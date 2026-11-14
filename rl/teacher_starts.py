"""The run teacher's fights as a fourth source of the fight loop's practice mix (D60 item 5, lane L).

    python rl/teacher_starts.py census --store STORE [--gens 14,15]                      starts by kind the store would give (no game)
    python rl/teacher_starts.py build  --root ROOT --store STORE [--gens 14,15] [--procs 2] [--limit N]    replay the runs, write starts/teacher-starts.*

The run teacher (`rl/run_teacher.py`, docs/lab/TEACHER_JOB.md) plays whole runs and stores each one: its seed, its RunRecord (every fight the
run entered, with the player's HP and deck at its first decision) and a record of every screen the teacher decided (`decisions.d/<seed>.jsonl`:
the screen, the player's own pick and the teacher's). The fight loop's practice (docs/rl/GYM_LOOP.md section 3) draws fights from the nights'
AI runs, from human runs and from the AI's own lost fights; none of them is the kind of fight the teacher's runs are actually in, with the
decks the teacher's choices build. This module makes those fights a fourth source, `teacher`, in the loop's own start format:

- **Which fights.** Every Act 1 elite, Act 1 boss, Act 2 elite and Act 2 boss (floors up to 33) a finished run entered, from the run's record
  (`record_fights`, `wanted_of`): not a hallway fight, an event's fight or an Act 3 fight. A start's id is `teacher:<seed>|f<floor>`, its
  `night` label `teacher-g<generation>`, its `source` `teacher`.
- **The replay.** The run is a function of its seed, its player (the generation's agent, named by the shard's manifest and checked by its
  `agent_sha256`) and the teacher's picks, all stored: `replay_run` plays the run again as `run_teacher.new_run` did (the player's pick is
  drawn at every screen, so its exploration stream runs as it did; the stored teacher pick is taken where the screen has a record), with the
  player's own fight search on the engine the teacher played on (`ENGINE`, the old engine: the teacher's job sets none), and at the first
  decision of each wanted fight exports the position as the nights' replay does (`native.get_state_json` on a fork of the fight). It stops at
  the last wanted fight. Every screen with a record and every fight start is checked against the run's own record (floor, options, HP, deck,
  the player's pick) and the replay stops at the first difference: the starts it made before it stand, the rest of that run is not used.
  Nothing is written to the store: records are read, never loaded through `RunFiles.load` (which cuts files).
- **Exclusions** (as the nights' source): a run behind the board's 200 Act-1 bosses (`gym_loop.board_run_seeds`) or listed in the exam's
  `exam/exclude_runs.json` is never replayed and never in the set; the sampler leaves them out again at every draw. The trainer's held-out
  split is by run seed (`is_val_seed`), as for every start.
- **Following the store.** `stale` says when the set must be built again (none yet, or `MIN_NEW` finished runs since the last build);
  the daemon checks it between steps (`Daemon.teacher_step`, `--teacher-store`, off by default), a run is replayed once (its row, with or
  without starts, is kept in `starts/teacher-replays.jsonl`), and the set is assembled again from the kept rows (`teacher-starts.jsonl`,
  its byte index, `teacher-starts.lite.json`, `teacher-meta.json`).
- **The mix.** `practice_mix` reads `teacher_share` per kind from `starts/mix.json` (none by default): the teacher takes that share of a
  kind's draws, the human share splits the rest.

Lab code: nothing here changes the engine, the search or the solver, and the teacher's store is only read.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time
from collections import Counter
from pathlib import Path
from typing import Any, Callable, Sequence

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fight_gym as fg  # noqa: E402 -- pure at import (gym_loop's rule: no native library until a worker selects its engine)
import gym_loop as gl  # noqa: E402
import practice_mix as pmix  # noqa: E402

STEM = "teacher-starts"  # the set's files under starts/: <stem>.jsonl, .index.json, .lite.json
REPLAYS = "teacher-replays.jsonl"  # one row per replayed run (starts or none): a run is replayed once
META = "teacher-meta.json"
GENERATIONS = (14, 15)
MIN_NEW = 25  # finished runs since the last build that make the set stale (the first build takes any number)
KIND_OF_NODE = {fg.NODE_ELITE: "elite", fg.NODE_BOSS: "boss"}  # the record's node types the loop wants
LAST_FLOOR = 33  # the Act 2 boss: the gym plays Acts 1 and 2
CATS = ("act1-elite", "act1-boss", "act2-elite", "act2-boss")
ENGINE = "old"  # the engine the teacher played on: `run_teacher.py job` sets none (STS2_ENGINE unset)
RUN_WALL_CAP_S = 1200.0  # a replay's own wall cap; the pool kills the worker `HARD_S` after it starts
HARD_S = 1500.0


# ---------------------------------------------------------------------------------------------------------------- the store (read only)


def _rows(path: Path) -> list[dict]:
    return list(gl.iter_jsonl(Path(path)))


def shard_manifests(gen_dir: Path) -> dict[str, dict]:
    """{job: manifest} of a generation's shards (a job's parts share one): who played it. A job with no shard yet has none."""
    out: dict[str, dict] = {}
    for p in sorted((Path(gen_dir) / "shards").glob("*/manifest.json")):
        m = gl.read_json(p, None)
        if isinstance(m, dict) and m.get("job") and m.get("agent") and m.get("agent_sha256"):
            out.setdefault(str(m["job"]), m)
    return out


def finished_runs(store: Path, generations: Sequence[int] = GENERATIONS) -> list[dict]:
    """The store's finished runs, a generation at a time in the order of its `runs.jsonl`: seed, ix, generation, job and the player's
    agent (`agent`, `agent_sha256`) and engine (the shard manifest's tag). A run is finished when its row says it ended (a win or a death),
    is not capped and has a record; one whose job has no shard yet (so no manifest to say who played) is not counted until it has."""
    out: list[dict] = []
    for g in generations:
        gen = Path(store) / str(g)
        mans = shard_manifests(gen)
        last: dict[str, dict] = {}
        for r in _rows(gen / "runs.jsonl"):
            if r.get("capped") or r.get("end") not in ("win", "death") or not r.get("seed"):
                continue
            last[str(r["seed"])] = r
        for seed, r in last.items():
            m = mans.get(str(r.get("job")))
            if m is None or not (gen / "decisions.d" / f"{seed}.record.json").is_file():
                continue
            out.append({"seed": seed, "ix": int(r.get("ix", -1)), "generation": int(g), "job": str(r["job"]), "agent": str(m["agent"]),
                        "agent_sha256": str(m["agent_sha256"]), "engine": m.get("engine")})
    return out


def record_fights(gen_dir: Path, seed: str) -> list[dict] | None:
    """Every fight the run entered, from its RunRecord, in order: floor, node type, act (1 up), and the player's HP, max HP and deck size at
    its first decision (what the replay's fight start is checked against). None when the run has no record."""
    rec = gl.read_json(Path(gen_dir) / "decisions.d" / f"{seed}.record.json", None)
    if not isinstance(rec, dict) or not isinstance(rec.get("fights"), list):
        return None
    out = []
    for f in rec["fights"]:
        st = f.get("state") or {}
        out.append({"floor": int(f["floor"]), "node_type": int(f["node_type"]), "act": int(st.get("act_index", 0)) + 1, "hp": int(st.get("hp", 0)),
                    "max_hp": int(st.get("max_hp", 0)), "deck": len(st.get("deck") or [])})
    return out


def wanted_of(fights: Sequence[dict], seed: str, generation: int, ix: int | None = None) -> list[dict]:
    """The starts a run's fights give: the Act 1 and Act 2 elite and boss fights, as the gym's start records begin (id, kind, act, floor, seed,
    run index, label); the replay adds the position."""
    out = []
    for f in fights:
        kind = KIND_OF_NODE.get(int(f["node_type"]))
        if kind is None or int(f["act"]) not in (1, 2) or int(f["floor"]) > LAST_FLOOR:
            continue
        out.append({"id": f"teacher:{seed}|f{int(f['floor']):02d}", "cat": f"act{int(f['act'])}-{kind}", "kind": kind, "act": int(f["act"]),
                    "floor": int(f["floor"]), "seed": seed, "ix": -1 if ix is None else int(ix), "night": f"teacher-g{generation}"})
    return out


def stored_screens(gen_dir: Path, seed: str) -> dict[int, dict]:
    """The screens the teacher decided, by the run's decision ordinal: what the replay takes the teacher's pick from and checks the screen
    against (type, floor, HP, number of options, the player's own pick). A refused screen keeps the player's pick as the teacher's."""
    out: dict[int, dict] = {}
    for r in _rows(Path(gen_dir) / "decisions.d" / f"{seed}.jsonl"):
        try:
            out[int(r["ordinal"])] = {"type": str(r["type"]), "floor": int(r["floor"]), "hp": int(r["hp"]), "max_hp": int(r["max_hp"]),
                                      "n_options": int(r["n_options"]), "player": int(r["player"]), "teacher": int(r.get("teacher", r["player"]))}
        except (KeyError, TypeError, ValueError):
            continue
    return out


def census(store: Path, generations: Sequence[int] = GENERATIONS) -> dict:
    """What the store would give, from the records alone (no game): finished runs, the starts by kind, the runs that have a start of each kind,
    and the same by generation."""
    starts: Counter = Counter()
    runs_with: Counter = Counter()
    by_gen: dict[str, Counter] = {}
    n = 0
    for run in finished_runs(store, generations):
        fights = record_fights(Path(store) / str(run["generation"]), run["seed"])
        if fights is None:
            continue
        n += 1
        w = wanted_of(fights, run["seed"], run["generation"])
        for cat in {x["cat"] for x in w}:
            runs_with[cat] += 1
        for x in w:
            starts[x["cat"]] += 1
            by_gen.setdefault(str(run["generation"]), Counter())[x["cat"]] += 1
    return {"runs": n, "starts": {c: starts[c] for c in CATS if starts[c]}, "runs_with": {c: runs_with[c] for c in CATS if runs_with[c]},
            "by_generation": {g: {c: v[c] for c in CATS if v[c]} for g, v in sorted(by_gen.items())}}


# ---------------------------------------------------------------------------------------------------------------- the replay


def screen_differs(have: dict, d: Any, view: Any, index: int) -> str | None:
    """Why a replayed screen is not the run's (None: it is): the record's type, floor, options and HP against the replay's, then the player's
    pick, which the run's own streams must reproduce."""
    v = d.view or view
    for name, mine, theirs in (("type", d.type, have["type"]), ("floor", int(v.floor), have["floor"]), ("options", len(d.options), have["n_options"]),
                               ("hp", int(v.hp), have["hp"]), ("max_hp", int(v.max_hp), have["max_hp"])):
        if mine != theirs:
            return f"{name} {mine} against {theirs}"
    if int(index) != int(have["player"]):
        return f"the player's pick {int(index)} against the run's {have['player']}"
    return None


def fight_differs(have: dict | None, info: Any) -> str | None:
    """Why a replayed fight start is not the run's: no such fight in the record, or the HP, max HP, deck or room type differ."""
    if have is None:
        return "the run's record has none"
    for name, mine, theirs in (("hp", int(info["hp"]), have["hp"]), ("max_hp", int(info["max_hp"]), have["max_hp"]),
                               ("deck", int(info["deck_size"]), have["deck"]), ("node_type", int(info["node_type"]), have["node_type"])):
        if mine != theirs:
            return f"{name} {mine} against {theirs}"
    return None


def make_start(w: dict, s: Any, doc: dict, info: dict, k: int) -> dict:
    """One start in the loop's record format (`fight_gym._replay_task`'s fields, the document included) plus its source."""
    pots = doc.get("potions") or {}
    held = sum(1 for x in (pots.get("slots") or [])[: int(pots.get("max_slots", 0))] if x)
    i = s.info
    act = int(s.act_index) + 1
    node_type = int(i["node_type"])
    return {**w, "act": act, "k": int(k), "node_type": node_type, "hp": int(i["hp"]), "max_hp": int(i["max_hp"]), "deck_size": int(i["deck_size"]),
            "potions": held, "relics": int(i.get("relic_count", 0)), "encounter_id": doc.get("encounter_id"), "turn": int(doc.get("turn", 0)),
            "doc": doc, "kind_same": KIND_OF_NODE.get(node_type) == w["kind"] and act == w["act"], "replay_same": True, "source": "teacher",
            "generation": info.get("generation"), "job": info.get("job")}


def replay_run(s: Any, player: Any, rng: Any, explore: Any, info: dict, *, combat_phase: Any, capture: Callable[[Any], dict],
               clock: Callable[[], float] = time.perf_counter, wall_cap: float = RUN_WALL_CAP_S) -> dict:
    """Play a stored teacher run again to its last wanted fight and keep the position at each wanted fight's first decision.

    The loop is `run_teacher.new_run`'s: at every screen the player's pick is drawn (the exploration stream is the run's), a screen with a
    stored record takes the teacher's stored pick (the replay is checked against the record first: `screen_differs`), the fights are the
    player's own (`play_fight`, the run's search stream). At each fight's first decision the fight start is checked against the run's record
    (`fight_differs`) and, when wanted, exported by `capture(session)` as a start. The replay stops at the first difference (`diverged`:
    the starts made before it stand), when the last wanted fight is captured (`end` stopped), or when the run ends.
    """
    t0 = clock()
    wanted = {int(w["floor"]): w for w in info["wanted"]}
    fights = {int(f["floor"]): f for f in info["fights"]}
    screens = info.get("screens") or {}
    got: dict[int, dict] = {}
    seen: set[int] = set()
    res: dict[str, Any] = {"end": "?", "checked": 0, "diverged": None}
    ordinal = moves = 0
    while True:
        if clock() - t0 > wall_cap:
            res["end"] = "timeout"
            break
        d = s.decision()
        if d is None and s.phase != combat_phase:
            res["end"] = "win" if s.won else "death"
            break
        if d is not None:
            view = s.view()
            index = int(player.policy.pick(view, d, explore).index)  # drawn at every screen, whether or not the teacher decided it
            have = screens.get(ordinal)
            if have is not None:
                why = screen_differs(have, d, view, index)
                if why:
                    res["diverged"] = f"screen {ordinal} ({d.type}): {why}"
                    break
                res["checked"] += 1
                index = int(have["teacher"])
            ordinal += 1
            s.apply(d.options[index])
            continue
        floor = int(s.info["floor"])
        if floor not in seen:
            seen.add(floor)
            why = fight_differs(fights.get(floor), s.info)
            if why:
                res["diverged"] = f"fight on floor {floor}: {why}"
                break
            res["checked"] += 1
            if floor in wanted:
                got[floor] = make_start(wanted[floor], s, capture(s), info, moves)
                if len(got) == len(wanted):
                    res["end"] = "stopped"
                    break
        moves += int(player.play_fight(s, rng, None))
    if res["diverged"]:
        res["end"] = "diverged"
    res.update(captured=len(got), wanted=len(wanted), starts=[got[f] for f in sorted(got)], moves=moves,
               same=res["diverged"] is None and len(got) == len(wanted), floor=int(s.info["floor"]))
    return res


_W: dict = {}


def init_worker(engine: str, agent_root: str, lib: str | None) -> None:
    """A replay process: BelowNormal, the teacher's engine, one thread."""
    if lib:
        os.environ["STS2_LIB_PATH"] = str(lib)
    os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")
    for name in ("OMP_NUM_THREADS", "MKL_NUM_THREADS"):
        os.environ.setdefault(name, "1")
    from sts2_gym import procutil  # noqa: PLC0415

    procutil.lower_priority()  # Windows priority class
    import gen0  # noqa: PLC0415

    gen0.select_engine(engine)
    _W.update(agent_root=Path(agent_root), players={})


def _player(agent: str):
    """The player of an agent folder (as the teacher's job built it: a3-v1 search, its own schedule, no judge: the judge reads boss doors,
    never a fight) and the agent's identity, once a process."""
    if agent not in _W["players"]:
        import gen_loop  # noqa: PLC0415
        import hindsight  # noqa: PLC0415
        import run_teacher as rt  # noqa: PLC0415

        d = _W["agent_root"] / agent
        player = hindsight.build_player(d, config="a3-v1", judge_path=None, own_schedule=True)
        _W["players"][agent] = (player, rt.agent_identity(d, gen_loop.load_agent(d))["agent_sha256"])
    return _W["players"][agent]


def replay_task(task: tuple) -> dict:
    """One run: `task` = ("teacher-replay", seed, ix, info). Returns the run's row (its starts included)."""
    import hindsight  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from run_train import engine as run_engine  # noqa: PLC0415
    from sts2_gym import native  # noqa: PLC0415

    _, seed, ix, info = task
    t0, c0 = time.perf_counter(), time.process_time()
    base = {"_task": [task[0], seed, ix], "seed": seed, "ix": ix, "generation": info.get("generation"), "job": info.get("job")}
    player, sha = _player(info["agent"])
    if sha != info["agent_sha256"]:
        return {**base, "skipped": "agent changed", "starts": []}
    rng = np.random.default_rng([hindsight.AGENT_SEED, int(ix)])
    explore = np.random.default_rng([hindsight.AGENT_SEED, int(ix), 1])
    s = hindsight.session_for(seed, player.agent, "a3-v1", own_schedule=True)

    def capture(sess: Any) -> dict:
        root = sess.combat(fork_combat=True).combat_fork(None)
        try:
            return native.get_state_json(root.handle)
        finally:
            native.destroy(root.handle) if not hasattr(root, "close") else root.close()

    try:
        res = replay_run(s, player, rng, explore, info, combat_phase=run_engine.rc.PHASE_COMBAT, capture=capture)
    finally:
        s.close()
    return {**base, **res, "seconds": round(time.perf_counter() - t0, 1), "cpu": round(time.process_time() - c0, 1)}


def _capped(task: tuple, reason: str, seconds: float, prog: dict) -> dict:
    return {"_task": [task[0], task[1], task[2]], "seed": task[1], "ix": task[2], "capped": True, "cap_reason": reason,
            "seconds": round(seconds, 1), "starts": []}


# ---------------------------------------------------------------------------------------------------------------- the set


def _code(row: dict) -> str:
    """What became of a run, in a few words (the meta's `done`)."""
    if row.get("capped"):
        return f"capped: {row.get('cap_reason')}"
    if row.get("skipped"):
        return f"skipped: {row['skipped']}"
    if row.get("diverged"):
        return f"diverged: {row['diverged']}"
    return "ok"


def _excluded_runs(loop: Any) -> frozenset[str]:
    ex = set(pmix.read_exclusion(loop.root))
    try:
        ex |= set(gl.board_run_seeds())
    except OSError:
        pass
    return frozenset(ex)


def _run_excluded(seed: str, ex: frozenset[str]) -> bool:
    return pmix.excluded({"id": f"teacher:{seed}|f00", "seed": seed, "run": seed}, ex)


def stale(loop: Any, store: Path, generations: Sequence[int] = GENERATIONS, min_new: int = MIN_NEW) -> str | None:
    """Why the set must be built again (None: it is current): the store has finished runs and there is no set, or `min_new` or more finished
    runs (not excluded) have come since the last build. A store that is not there has nothing to follow."""
    if not Path(store).is_dir():
        return None
    ex = _excluded_runs(loop)
    runs = [r for r in finished_runs(store, generations) if not _run_excluded(r["seed"], ex)]
    meta = gl.read_json(Path(loop.starts) / META, None)
    if not runs:
        return None
    if not meta or not (Path(loop.starts) / f"{STEM}.jsonl").is_file():
        return f"no teacher start set ({len(runs)} runs)"
    done = meta.get("done") or {}
    new = [r for r in runs if r["seed"] not in done]
    return f"{len(new)} new teacher runs" if len(new) >= max(1, int(min_new)) else None


def build(loop: Any, store: Path, generations: Sequence[int] = GENERATIONS, *, procs: int = 6, agent_root: Path | None = None, lib: str | None = None,
          engine: str = ENGINE, replay: Callable[[tuple], dict] | None = None, limit: int | None = None, stop: Callable[[], str | None] | None = None,
          quiet: bool = True) -> dict:
    """Replay the store's finished runs that have not been (each once; the rows are kept in `starts/teacher-replays.jsonl`) and assemble the
    set (`teacher-starts.jsonl`, its index, its lite list, `teacher-meta.json`) from the rows of the runs not excluded. `replay` runs a task
    in this process (the tests'); else a spawn pool of `procs` BelowNormal processes plays them under the daemon's QUIET line and `stop`'s
    reasons (`quiet` False: no QUIET.lock line, for a hand-run trial; `stop` then says when to stop). `limit` replays only that many new runs
    (a seeded spread over the store: a sample). Returns the meta."""
    store = Path(store)
    sd = Path(loop.starts)
    target = sd / REPLAYS
    ex = _excluded_runs(loop)
    runs = finished_runs(store, generations)
    kept = [r for r in runs if not _run_excluded(r["seed"], ex)]
    excluded = len(runs) - len(kept)
    done = {str(r.get("seed")) for r in gl.iter_jsonl(target)}
    todo = [r for r in kept if r["seed"] not in done]
    if limit is not None:
        import hashlib  # noqa: PLC0415

        todo = sorted(todo, key=lambda r: hashlib.sha256(r["seed"].encode()).hexdigest())[: int(limit)]
    tasks, direct = [], []
    for r in todo:
        gen = store / str(r["generation"])
        fights = record_fights(gen, r["seed"])
        wanted = wanted_of(fights or [], r["seed"], r["generation"], r["ix"])
        base = {"_task": ["teacher-replay", r["seed"], r["ix"]], "seed": r["seed"], "ix": r["ix"], "generation": r["generation"], "job": r["job"],
                "starts": []}
        if fights is None or not wanted:  # nothing to replay for: recorded, so it is not asked again
            direct.append({**base, "skipped": "no record" if fights is None else "no wanted fight"})
            continue
        tasks.append(("teacher-replay", r["seed"], r["ix"],
                      {"agent": r["agent"], "agent_sha256": r["agent_sha256"], "generation": r["generation"], "job": r["job"], "engine": r["engine"],
                       "wanted": wanted, "fights": fights, "screens": stored_screens(gen, r["seed"])}))
    stopped = None
    with open(target, "a", encoding="utf-8") as fh:
        for row in direct:
            fh.write(json.dumps(row) + "\n")
    if tasks:
        if replay is not None:
            with open(target, "a", encoding="utf-8") as fh:
                for t in tasks:
                    fh.write(json.dumps(replay(t)) + "\n")
        else:
            agent_root = Path(agent_root) if agent_root else gl.REF / "agent"

            def play() -> str | None:
                return fg.run_tasks(tasks, replay_task, target, procs, init=init_worker, init_args=(engine, str(agent_root), lib),
                                    hard_of=lambda t: HARD_S, capped_of=_capped, mem_limit=fg.MEM_LIMIT_GB * 2**30,
                                    stop=stop or (lambda: gl.why_blocked(loop.root, starting=False)))

            stopped = gl.guarded(loop, play) if quiet else play()
    codes: dict[str, str] = {}
    for r in gl.iter_jsonl(target):
        if r.get("seed"):
            codes[str(r["seed"])] = _code(r)
    for r in runs:
        if _run_excluded(r["seed"], ex):
            codes[r["seed"]] = "excluded"
    seeds = {r["seed"] for r in kept}
    lite = gl.assemble_starts(sd, [(target, seeds)], stem=STEM)
    meta = {"spec": {"generations": [int(g) for g in generations], "engine": engine}, "done": codes, "starts": len(lite),
            "by_cat": {c: sum(1 for s in lite if s["cat"] == c) for c in CATS if any(s["cat"] == c for s in lite)},
            "by_split": dict(Counter(s["split"] for s in lite)), "excluded": excluded,
            "runs": {"finished": len(runs), "replayed": sum(1 for r in kept if r["seed"] in codes), "ok": sum(1 for c in codes.values() if c == "ok"),
                     "by_outcome": dict(Counter(c.split(":")[0] for c in codes.values()))},
            "written": gl.stamp()}
    if stopped:
        meta["stopped"] = stopped
    gl.write_json(sd / META, meta)
    loop.log(f"teacher starts: {meta['starts']} starts {meta['by_cat']} from {meta['runs']['replayed']} runs "
             f"(ok {meta['runs']['ok']}, outcomes {meta['runs']['by_outcome']}, excluded {excluded})" + (f"; stopped ({stopped})" if stopped else ""))
    return meta


# ---------------------------------------------------------------------------------------------------------------- commands


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("census", "build"):
        p = sub.add_parser(name)
        p.add_argument("--store", required=True)
        p.add_argument("--gens", default=",".join(str(g) for g in GENERATIONS))
        if name == "build":
            p.add_argument("--root", required=True, help="the loop's folder (its starts/ gets the files); a scratch folder for a trial")
            p.add_argument("--procs", type=int, default=2)
            p.add_argument("--limit", type=int, default=None, help="replay only this many new runs (a spread over the store)")
            p.add_argument("--engine", default=ENGINE)
            p.add_argument("--lib", default=None, help="the native library folder (default: this tree's out/)")
            p.add_argument("--agent-root", default=str(gl.REF / "agent"))
    a = ap.parse_args(argv)
    gens = tuple(int(x) for x in a.gens.split(",") if x.strip())
    if a.cmd == "census":
        print(json.dumps(census(Path(a.store), gens), indent=1))
        return 0
    loop = gl.Loop(a.root)

    def hand_stop() -> str | None:  # a trial by hand holds no QUIET line: it yields to HALT, PAUSE and the night only
        return "halt" if gl.HALT.exists() else "pause" if gl.PAUSE.exists() else "night" if gl.night_on() else None

    meta = build(loop, Path(a.store), gens, procs=a.procs, agent_root=Path(a.agent_root), lib=a.lib, engine=a.engine, limit=a.limit,
                 stop=hand_stop, quiet=False)
    print(json.dumps({k: v for k, v in meta.items() if k != "done"}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
