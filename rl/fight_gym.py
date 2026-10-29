"""The fight gym: fight starts from a night's exact replays, played again under a search arm with fresh draws (M2, M3).

    python rl/fight_gym.py starts --source ARM --out OUT --act1-boss 200 --act2 200 --seed S
    python rl/fight_gym.py play   --out OUT --arm a3-v1|x4|x16 --seeds 10 [--cat act1-boss|act2] [--limit N]
    python rl/fight_gym.py m3     --out OUT [--seeds 10]
    python rl/fight_gym.py run    --out OUT --plan a3-v1:act1-boss,x16:act1-boss,x4:act1-boss,m3 --seeds 10
    python rl/fight_gym.py report --out OUT

`starts` replays the arm's runs exactly (`gen0.play_emu_run` with the agent of the night's generation, as
`tools/solver/coverage.py` does) and writes one `sts2-emu-snapshot/1` document per start at the fight's first combat
decision (the opening hand and first intents as dealt there), with its metadata, to OUT/starts.jsonl. Each replay ends at
its last wanted start, and is checked against the arm on the run so far: the floor, the node kinds, the map path, the
cards taken and the rest choices up to the start.

`play` plays every start `--seeds` times under an arm: the document is loaded and forked with fresh hidden information
(draw order and streams) from seed j, then each decision is `live_search.search_live` on the current handle and the best
move is stepped, to the fight's end. Seed j and every per-decision search seed derive from (start, j, decision index)
only, never from the arm, so the arms are paired. The arms are a3-v1 and a3-v1 with every simulation count times 4 and
times 16 (base, `high` and `hi`, so nothing clamps; the safety clock scales too); the prior is the agent's.

`m3` scores two predictions of a horizon-0 leaf against the gym's outcome from it. The leaf is where the solver puts it
(`Expectimax.MoveValueOf`: the position at which the player ends the turn, before the end-turn step), taken in each Act-1
boss start's a3-v1 game with j = 0 at the end of the player's first turn. The target is ten fresh-seed games from that
position under a3-v1 (End Turn stepped first), as a win rate and a mean L0.

Every job is resumable (rows are appended, done ones skipped) and holds the lab's night and lock guard: it stops its own
pool when the night is active or due, when main.lock exists, or when another live QUIET.lock line appears, and resumes when
the machine is free. Workers run at BelowNormal, at most six. Lab code: nothing here changes the engine or the solver.
"""

from __future__ import annotations

import argparse
import dataclasses
import datetime as dt
import hashlib
import json
import math
import multiprocessing as mp
import os
import random
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

REF = Path(os.environ.get("FIGHT_GYM_REFEREE", "D:/spireline/referee"))
QUIET = REF / "QUIET.lock"
MAIN_LOCK = REF / "backlog" / "main.lock"
ACTIVE = REF / "LAB" / "nightly" / "active"
TAG = "gym-m2-play"
NIGHT_WINDOW = (dt.time(3, 10), dt.time(3, 45))  # the night is due 03:23Z: never compute across its start
MAX_PROCS = 6
NODE_ELITE, NODE_BOSS = 2, 6  # run_train.vocab
ACT1_BOSS_INDEX = 15  # the run row's `nodes` string: index i is floor i + 2; the Act-1 boss is floor 17
ACT2_INDICES = range(16, 32)  # floors 18..33; the Act-2 boss is floor 33
ARMS = {"a3-v1": 1, "x4": 4, "x16": 16}
POTION_VALUE = 10.0  # live_search's and L0's potion value
DECISION_CAP = 400  # a game still going after this many decisions is counted as capped (and not won)
M3_PLAYOUTS = 30


# ---------------------------------------------------------------------------------------------------------- pure parts


def game_seed(start: str, j: int, what: int | str) -> int:
    """The seed of game j of a start, for `what` (the load, the fork, or decision k's search). Never depends on the arm,
    so every arm plays the same worlds with the same search seeds."""
    h = hashlib.sha256(f"{start}|{j}|{what}".encode()).digest()
    return int.from_bytes(h[:4], "little") % (1 << 30)


def l0(won: bool, hp: int, potions: int, max_hp: int) -> float:
    """L0: 0 if dead, else 0.5 + 0.5 * min(1, (HP + 10 * potions) / max HP)."""
    if not won:
        return 0.0
    return 0.5 + 0.5 * min(1.0, (hp + POTION_VALUE * potions) / max(1, max_hp))


def _eligible(runs: list[dict]) -> list[dict]:
    return sorted((r for r in runs if not r.get("crash") and r.get("end") in ("death", "win") and r.get("nodes")),
                  key=lambda r: int(r["ix"]))


def select_starts(runs: list[dict], n_act1_boss: int, n_act2: int, seed: int) -> list[dict]:
    """Seeded fight starts from an arm's runs: Act-1 boss fights from runs whose floor shows they reached the boss, and Act-2
    elite and boss fights (half each where there are enough, the rest from the other kind) from runs that reached them.
    Each start: {id, cat, kind, act, floor, seed, ix}; the order within a category is the seeded draw's, so its first n are
    a random subset."""
    rnd = random.Random(seed)
    ok = _eligible(runs)

    def start(r: dict, index: int, cat: str, kind: str, act: int) -> dict:
        floor = index + 2
        return {"id": f"{r['seed']}|f{floor:02d}", "cat": cat, "kind": kind, "act": act, "floor": floor, "seed": r["seed"],
                "ix": int(r["ix"])}

    boss1 = [start(r, ACT1_BOSS_INDEX, "act1-boss", "boss", 1) for r in ok
             if int(r["floor"]) >= ACT1_BOSS_INDEX + 2 and len(r["nodes"]) > ACT1_BOSS_INDEX and r["nodes"][ACT1_BOSS_INDEX] == "B"]
    elite2, boss2 = [], []
    for r in ok:
        for i in ACT2_INDICES:
            if i < len(r["nodes"]) and i + 2 <= int(r["floor"]) and r["nodes"][i] in "EB":
                (boss2 if r["nodes"][i] == "B" else elite2).append(
                    start(r, i, "act2-boss" if r["nodes"][i] == "B" else "act2-elite", "boss" if r["nodes"][i] == "B" else "elite", 2))
    pick1 = rnd.sample(boss1, min(n_act1_boss, len(boss1)))
    nb = min(len(boss2), n_act2 // 2)
    ne = min(len(elite2), n_act2 - nb)
    nb = min(len(boss2), n_act2 - ne)
    picks_b, picks_e = rnd.sample(boss2, nb), rnd.sample(elite2, ne)
    act2 = []
    for i in range(max(nb, ne)):  # interleaved, so any prefix holds both kinds
        act2 += [x[i] for x in (picks_b, picks_e) if i < len(x)]
    return pick1 + act2


def arm_settings(name: str) -> dict:
    """The arm as numbers: a3-v1's simulation counts (base, `high`, `hi`) and safety clock times the arm's factor."""
    import search_configs  # noqa: PLC0415 -- pure

    base = search_configs.get("a3-v1")
    m = ARMS[name]
    if m == 1:
        return {"cfg": base, "mult": 1}
    sch = dataclasses.replace(base.schedule, high=base.schedule.high * m, hi=base.schedule.hi * m)
    cfg = dataclasses.replace(base, name=name, sims=base.sims * m, time_budget_s=base.time_budget_s * m, schedule=sch)
    return {"cfg": cfg, "mult": m}


def done_keys(rows: list[dict], key) -> set:
    return {key(r) for r in rows}


def game_key(r: dict) -> tuple:
    return (r["arm"], r["start"], int(r["j"]))


def todo_games(starts: list[dict], arm: str, seeds: int, done: set) -> list[tuple[str, int]]:
    """(start id, j) pairs of an arm still to play, in start order then j."""
    return [(s["id"], j) for s in starts for j in range(seeds) if (arm, s["id"], j) not in done]


def mean_se(xs: list[float]) -> tuple[float | None, float | None, int]:
    """Mean and standard error (sample sd / sqrt n)."""
    n = len(xs)
    if not n:
        return None, None, 0
    m = sum(xs) / n
    if n < 2:
        return m, None, n
    var = sum((x - m) ** 2 for x in xs) / (n - 1)
    return m, math.sqrt(var / n), n


def paired_diff(rows: list[dict], arm: str, base: str, field: str = "won") -> dict:
    """arm minus base on the (start, j) games both played: the mean difference with two standard errors, over the games
    (pairs) and over the per-start mean differences (starts with the same js in both)."""
    a = {(r["start"], int(r["j"])): float(r[field]) for r in rows if r["arm"] == arm}
    b = {(r["start"], int(r["j"])): float(r[field]) for r in rows if r["arm"] == base}
    common = sorted(set(a) & set(b))
    d = [a[k] - b[k] for k in common]
    m, se_pairs, n = mean_se(d)
    per: dict[str, list[float]] = defaultdict(list)
    for k, x in zip(common, d):
        per[k[0]].append(x)
    ms, se_starts, ns = mean_se([sum(v) / len(v) for v in per.values()])
    return {"diff": m, "se_pairs": se_pairs, "pairs": n, "se_starts": se_starts, "starts": ns}


def _ranks(xs: list[float]) -> list[float]:
    order = sorted(range(len(xs)), key=lambda i: xs[i])
    r = [0.0] * len(xs)
    i = 0
    while i < len(order):
        k = i
        while k + 1 < len(order) and xs[order[k + 1]] == xs[order[i]]:
            k += 1
        for t in range(i, k + 1):
            r[order[t]] = (i + k) / 2.0 + 1.0
        i = k + 1
    return r


def pearson(x: list[float], y: list[float]) -> float | None:
    n = len(x)
    if n < 2:
        return None
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sx = math.sqrt(sum((a - mx) ** 2 for a in x))
    sy = math.sqrt(sum((b - my) ** 2 for b in y))
    return sxy / (sx * sy) if sx > 0 and sy > 0 else None


def spearman(x: list[float], y: list[float]) -> float | None:
    return pearson(_ranks(x), _ranks(y))


def mse(x: list[float], y: list[float]) -> float | None:
    return sum((a - b) ** 2 for a, b in zip(x, y)) / len(x) if x else None


def score_predictions(target: list[float], preds: dict[str, list[float]], boot: int = 2000, seed: int = 0) -> dict:
    """Pearson, Spearman and MSE of each prediction against the target, with 95% bootstrap intervals over positions, and
    for each metric the share of resamples in which each predictor is the better one."""
    rnd = random.Random(seed)
    n = len(target)
    metrics = {"pearson": pearson, "spearman": spearman, "mse": mse}
    out = {name: {m: f(p, target) for m, f in metrics.items()} for name, p in preds.items()}
    samples = {name: {m: [] for m in metrics} for name in preds}
    names = list(preds)
    wins = {m: dict.fromkeys(names, 0) for m in metrics}
    for _ in range(boot if n >= 3 else 0):
        idx = [rnd.randrange(n) for _ in range(n)]
        t = [target[i] for i in idx]
        vals = {}
        for name, p in preds.items():
            q = [p[i] for i in idx]
            vals[name] = {m: f(q, t) for m, f in metrics.items()}
            for m in metrics:
                if vals[name][m] is not None:
                    samples[name][m].append(vals[name][m])
        for m in metrics:
            got = {k: v[m] for k, v in vals.items() if v[m] is not None}
            if len(got) == len(names):
                best = (min if m == "mse" else max)(got, key=got.get)
                wins[m][best] += 1
    for name in preds:
        for m in metrics:
            s = sorted(samples[name][m])
            out[name][m + "_ci"] = [s[int(0.025 * (len(s) - 1))], s[int(0.975 * (len(s) - 1))]] if s else None
    out["better_share"] = {m: {k: v / max(1, sum(w.values())) for k, v in w.items()} for m, w in wins.items()}
    out["positions"] = n
    return out


# ---------------------------------------------------------------------------------------------------------- the guard


def utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def _age_h(line: str) -> float:
    try:
        stamp = line.split()[1].rstrip("Z")
        t = dt.datetime.fromisoformat(stamp if len(stamp) > 16 else stamp + ":00").replace(tzinfo=dt.timezone.utc)
        return (utc() - t).total_seconds() / 3600
    except (IndexError, ValueError):
        return 0.0


def _quiet_lines() -> list[str]:
    try:
        return [x for x in QUIET.read_text(encoding="utf-8").splitlines() if x.strip()]
    except FileNotFoundError:
        return []


def blocked() -> str | None:
    """Why the job must not compute now, or None: the night (active or due), main.lock, or another live QUIET line."""
    if ACTIVE.exists():
        return "nightly-active"
    if MAIN_LOCK.exists():
        return "main.lock"
    now = utc().time()
    if NIGHT_WINDOW[0] <= now < NIGHT_WINDOW[1]:
        return "night-window"
    for x in _quiet_lines():
        if x.startswith(TAG + " ") or x.startswith("w3-blocks"):
            continue
        if _age_h(x) < 2.0:
            return "quiet:" + x.split()[0]
    return None


def hold(on: bool) -> None:
    """Hold (or drop) this job's own QUIET.lock line; other lines are kept as they are."""
    keep = [x for x in _quiet_lines() if not x.startswith(TAG + " ")]
    if on:
        keep.append(f"{TAG} {utc().strftime('%Y-%m-%dT%H:%MZ')} pid={os.getpid()}")
    if keep:
        QUIET.write_text("".join(x + "\n" for x in keep), encoding="utf-8", newline="\n")
    else:
        QUIET.unlink(missing_ok=True)


def read_jsonl(p: Path) -> list[dict]:
    if not p.is_file():
        return []
    out = []
    with open(p, encoding="utf-8") as fh:
        for x in fh:
            if x.strip():
                try:
                    out.append(json.loads(x))
                except ValueError:
                    pass  # a line cut by a stop: its game is played again
    return out


def _say(out: Path, msg: str) -> None:
    with open(out / "gym.log", "a", encoding="utf-8") as fh:
        fh.write(f"{utc().strftime('%Y-%m-%dT%H:%M:%SZ')} {msg}\n")


def guarded_run(out: Path, what: str, tasks: list, fn, target: Path, procs: int, init_args: tuple) -> None:
    """Run `fn` over `tasks` in a spawn pool (this job's own children only), appending each result to `target`; stop the
    pool whenever `blocked()` and resume when free. The caller filters done tasks before; tasks done here are skipped on a
    resume by the same filter."""
    left = list(tasks)
    while left:
        why = blocked()
        if why:
            _say(out, f"{what}: waiting ({why}); {len(left)} left")
            while blocked():
                time.sleep(60)
        hold(True)
        _say(out, f"{what}: computing; {len(left)} left")
        pool = mp.get_context("spawn").Pool(processes=min(MAX_PROCS, max(1, procs)), initializer=_init_worker,
                                            initargs=init_args, maxtasksperchild=40)
        stopped, refreshed, finished = None, time.time(), []
        try:
            it = pool.imap_unordered(fn, left)
            with open(target, "a", encoding="utf-8") as fh:
                for _ in range(len(left)):
                    while True:
                        try:
                            res = it.next(timeout=10)
                            break
                        except mp.TimeoutError:
                            stopped = blocked()
                            if stopped:
                                break
                    if stopped:
                        break
                    fh.write(json.dumps(res) + "\n")
                    fh.flush()
                    finished.append(res.get("_task"))
                    if time.time() - refreshed > 3000:
                        hold(True)
                        refreshed = time.time()
                    stopped = blocked()
                    if stopped:
                        break
        finally:
            pool.terminate()  # this pool's own children only
            pool.join()
            hold(False)
        gone = {json.dumps(t) for t in finished}
        left = [t for t in left if json.dumps(_task_id(t)) not in gone]
        if stopped:
            _say(out, f"{what}: stopped ({stopped}); {len(left)} left")
    _say(out, f"{what}: done")


def _task_id(task: tuple) -> list:
    return [task[0], task[1], task[2]]


# ---------------------------------------------------------------------------------------------------------- workers

_W: dict = {}


def _init_worker(engine: str, agent_root: str, agent_dir: str, out: str) -> None:
    sys.path.insert(0, str(ROOT))
    from tools.solver import coverage  # noqa: PLC0415

    coverage._below_normal()  # noqa: SLF001 -- Windows priority class
    import gen0  # noqa: PLC0415

    gen0.select_engine(engine)
    import gen_loop  # noqa: PLC0415

    _W["agent"] = gen_loop.load_agent(Path(agent_root) / agent_dir)
    _W["agent_dir"] = Path(agent_root) / agent_dir
    _W["out"] = Path(out)
    _W["starts"] = None


def _starts_by_id() -> dict:
    if _W.get("starts") is None:
        _W["starts"] = {s["id"]: s for s in read_jsonl(_W["out"] / "starts.jsonl")}
    return _W["starts"]


def _replay_task(task: tuple) -> dict:
    """Replay one run of the arm to its last wanted start; return the starts' documents and the replay check."""
    _, run_seed, ix, config, wanted, arm_row = task
    import gen0  # noqa: PLC0415
    from sts2_gym import native  # noqa: PLC0415

    floors = {int(w["floor"]): w for w in wanted}
    last = max(floors)
    got: dict[int, dict] = {}

    def hook(k, root, s, pos):
        f = int(s.info["floor"])
        if f in floors and f not in got:
            doc = native.get_state_json(root.handle)
            pots = doc.get("potions") or {}
            held = sum(1 for x in (pots.get("slots") or [])[: int(pots.get("max_slots", 0))] if x)
            got[f] = {"k": k, "node_type": int(s.info["node_type"]), "act": int(s.act_index) + 1, "hp": int(s.info["hp"]),
                      "max_hp": int(s.info["max_hp"]), "deck_size": int(s.info["deck_size"]), "potions": held,
                      "relics": int(s.info.get("relic_count", 0)), "encounter_id": doc.get("encounter_id"),
                      "turn": int(doc.get("turn", 0)), "doc": doc}
        return f > last or len(got) == len(floors)

    t0 = time.perf_counter()
    r = gen0.play_emu_run(run_seed, ix, config, agent=_W["agent"], hook=hook)
    nodes = r.get("nodes") or ""
    path_same = arm_row["nodes"][: len(nodes)] == nodes and len(nodes) == last - 1
    cards_same = [list(c) for c in r.get("cards") or []] == [list(c) for c in arm_row.get("cards") or [] if int(c[0]) < last]
    rests_same = [list(c) for c in r.get("rests") or []] == [list(c) for c in arm_row.get("rests") or [] if int(c[0]) < last]
    starts = []
    for f, w in sorted(floors.items()):
        g = got.get(f)
        if g is None:
            continue
        want_type = NODE_BOSS if w["kind"] == "boss" else NODE_ELITE
        starts.append({**w, **g, "kind_same": g["node_type"] == want_type and g["act"] == w["act"]})
    same = path_same and cards_same and rests_same and len(got) == len(floors) and all(s["kind_same"] for s in starts)
    return {"_task": [task[0], run_seed, ix], "seed": run_seed, "ix": ix, "wanted": len(floors), "captured": len(got),
            "floor": int(r["floor"]), "end": r.get("end"), "path_same": path_same, "cards_same": cards_same,
            "rests_same": rests_same, "same": same, "seconds": round(time.perf_counter() - t0, 1),
            "starts": [{**s, "replay_same": same} for s in starts]}


def _fresh_world(doc: dict, start: str, j: int, ns: str):
    """The start's position with fresh hidden information from game j's seeds: loaded, then a resampled fork."""
    import combat_mcts  # noqa: PLC0415
    from sts2_gym import native  # noqa: PLC0415

    base = combat_mcts.Combat(native.load_combat_snapshot(doc, game_seed(start, j, ns + "load")))
    try:
        world = base.fork(game_seed(start, j, ns + "fork"))
    finally:
        base.close()
    world.encounter_id = native.encounter_id(world.handle)
    return world


def _play(world, start: dict, j: int, cfg, extra: dict, ns: str, capture: bool = False) -> dict:
    """Play the world to the fight's end under the search config; `capture` keeps the end of the first player turn."""
    import live_search  # noqa: PLC0415
    import search_configs  # noqa: PLC0415
    from sts2_gym import native  # noqa: PLC0415

    t0 = time.perf_counter()
    k = sims = potions_used = end_turns = exhausted = assumed = 0
    first: dict = {}
    leaf = None
    max_search = 0.0
    while not world.done and k < DECISION_CAP:
        info = {"node_type": start["node_type"], "floor": start["floor"], "max_hp": world.max_hp()}
        pos = search_configs.position_of(world, info)
        kw = cfg.search_kwargs(pos, env_prior=False)
        kw.update(extra)
        res = live_search.search_live(world.handle, seed=game_seed(start["id"], j, f"{ns}{k}"), **kw)
        sims += res.sims
        exhausted += bool(res.budget_exhausted)
        assumed += bool(res.ai_state_assumed)
        max_search = max(max_search, res.elapsed_s)
        if k == 0:
            vis = sum(c.visits for c in res.children)
            dead = sum(c.deaths for c in res.children)
            first = {"root_value": (sum(c.value_sum for c in res.children) / vis) if vis else None,
                     "root_p_death": dead / vis if vis else None, "root_best_value": None if math.isnan(res.best.value_mean) else res.best.value_mean,
                     "root_sims": res.sims, "first_move": [res.best.action, res.best.target, res.best.kind]}
        best = res.best
        if best.kind == "potion":
            potions_used += 1
        if best.kind == "end_turn":
            if capture and end_turns == 0 and leaf is None:
                leaf = native.get_state_json(world.handle)
            end_turns += 1
        live_search._step(world, best.action, best.target)  # noqa: SLF001
        k += 1
    won = bool(world.done and world.won())
    out = live_search._outcome(world, POTION_VALUE)  # noqa: SLF001
    held = int(round((out.value - out.hp) / POTION_VALUE)) if not out.died else 0
    row = {"won": won, "end_hp": world.hp() if won else 0, "max_hp": world.max_hp(), "potions_used": potions_used,
           "potions_held": held, "turns": end_turns + 1, "decisions": k, "sims": sims, "capped": not world.done,
           "budget_exhausted": exhausted, "ai_assumed": assumed, "max_search_s": round(max_search, 3),
           "L0": l0(won, world.hp(), held, world.max_hp()), "seconds": round(time.perf_counter() - t0, 2), **first}
    if capture:
        row["_leaf"] = leaf
    return row


def _game_task(task: tuple) -> dict:
    arm, start_id, j = task
    start = _starts_by_id()[start_id]
    cfg = arm_settings(arm)["cfg"]
    extra = _W["agent"].search_extra()
    world = _fresh_world(start["doc"], start_id, j, "g")
    try:
        row = _play(world, start, j, cfg, extra, "g", capture=arm == "a3-v1" and j == 0 and start["cat"] == "act1-boss")
    finally:
        world.close()
    return {"_task": [arm, start_id, j], "arm": arm, "start": start_id, "j": j, "cat": start["cat"], **row}


def _value_head(handle: int) -> dict:
    """The agent's net's value head on the position: P(win) and its L0 mapping (end-HP buckets at their centres)."""
    import ctypes  # noqa: PLC0415

    import combat_prior  # noqa: PLC0415
    import numpy as np  # noqa: PLC0415
    from combat_tokens import bucket_centres, encode  # noqa: PLC0415
    from sts2_gym import native  # noqa: PLC0415

    if "net" not in _W:
        import torch  # noqa: PLC0415

        path = _W["agent_dir"] / "combat_net.pt"
        _W["net"] = combat_prior.PolicyNet(path)
        blob = torch.load(str(path), map_location="cpu", weights_only=True)
        _W["centres"] = bucket_centres(int(blob["n_buckets"]), int(blob["width"]))
    net = _W["net"]
    obs = (ctypes.c_int * native.OBS_SIZE)()
    view = native.clone(handle, False, 0, obs)
    native.destroy(view)
    tokens = encode(handle, obs, native.encounter_id(handle), probe=False)
    with net._torch.inference_mode():  # noqa: SLF001
        value_logits, _ = net.model(**net._batch([tokens]))  # noqa: SLF001
    p = np.exp(value_logits[0].numpy() - value_logits[0].numpy().max())
    p = p / p.sum()
    max_hp = max(1, int(obs[1]))
    c = _W["centres"]
    lmap = float(sum(p[k] * (0.5 + 0.5 * min(1.0, float(c[k]) / max_hp)) for k in range(1, len(p))))
    return {"vh_win": float(1.0 - p[0]), "vh_L0": lmap, "vh_hp": float(sum(p[k] * c[k] for k in range(1, len(p))))}


def _m3_task(task: tuple) -> dict:
    """("pred", start, 0): the two predictions at the leaf; ("cont", start, j): one fresh-seed a3-v1 game from it."""
    kind, start_id, j = task
    import combat_mcts  # noqa: PLC0415
    import live_search  # noqa: PLC0415
    from sts2_gym import native  # noqa: PLC0415

    leaf = {r["start"]: r for r in read_jsonl(_W["out"] / "m3_leaves.jsonl")}[start_id]
    start = _starts_by_id()[start_id]
    if kind == "pred":
        root = combat_mcts.Combat(native.load_combat_snapshot(leaf["doc"], game_seed(start_id, 0, "m3pred")))
        try:
            view = root.fork(None)
            row = _value_head(view.handle)
            children, max_hp = live_search._root_moves(view)  # noqa: SLF001
            end = next(c for c in children if c.node.kind == "end_turn")
            outs = [live_search._playout(view, end, game_seed(start_id, i, "m3po"), POTION_VALUE) for i in range(M3_PLAYOUTS)]  # noqa: SLF001
            outs = [o for o in outs if o is not None]
            vals = [0.0 if o.died else 0.5 + 0.5 * min(1.0, o.value / max(1, max_hp)) for o in outs]
            row.update({"po_win": sum(not o.died for o in outs) / len(outs), "po_L0": sum(vals) / len(vals),
                        "po_hp_value": sum(o.value for o in outs) / len(outs), "po_n": len(outs),
                        "po_truncated": sum(o.truncated for o in outs), "leaf_hp": view.hp(), "leaf_max_hp": max_hp})
            view.close()
        finally:
            root.close()
        return {"_task": [kind, start_id, j], "type": kind, "start": start_id, "j": j, **row}
    cfg = arm_settings("a3-v1")["cfg"]
    world = _fresh_world(leaf["doc"], start_id, j, "m3")
    try:
        end = next(m for m in live_search._world_moves(world) if m.kind == "end_turn")  # noqa: SLF001
        live_search._step(world, end.action, end.target)  # noqa: SLF001
        row = _play(world, start, j, cfg, _W["agent"].search_extra(), "m3") if not world.done else {
            "won": bool(world.won()), "L0": l0(bool(world.won()), world.hp(), 0, world.max_hp())}
    finally:
        world.close()
    return {"_task": [kind, start_id, j], "type": kind, "start": start_id, "j": j, **row}


# ---------------------------------------------------------------------------------------------------------- commands


def _meta(out: Path) -> dict:
    return json.loads((out / "meta.json").read_text(encoding="utf-8"))


def _init_args(out: Path) -> tuple:
    m = _meta(out)
    return (m["engine"], m["agent_root"], m["agent"], str(out))


def cmd_starts(a) -> int:
    sys.path.insert(0, str(ROOT))
    from tools.solver import coverage  # noqa: PLC0415

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    summ, runs, agent_dir, engine = coverage._arm(a.source)  # noqa: SLF001
    picks = select_starts(runs, a.act1_boss, a.act2, a.seed)
    meta = {"source": coverage.source_label(a.source), "engine": engine, "agent": agent_dir, "agent_root": a.agent_root,
            "config": summ.get("config") or "a3-v1", "seed": a.seed, "act1_boss": a.act1_boss, "act2": a.act2,
            "picked": len(picks), "start_point": "the fight's first combat decision (opening hand and first intents as dealt)"}
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    by_run: dict[str, list[dict]] = defaultdict(list)
    for p in picks:
        by_run[p["seed"]].append(p)
    rows = {r["seed"]: r for r in runs}
    done = {r["seed"] for r in read_jsonl(out / "replays.jsonl")}
    tasks = [("replay", sd, int(ws[0]["ix"]), meta["config"], ws, rows[sd]) for sd, ws in by_run.items() if sd not in done]
    guarded_run(out, "starts", tasks, _replay_task, out / "replays.jsonl", a.procs, _init_args(out))
    reps = read_jsonl(out / "replays.jsonl")
    order = {p["id"]: i for i, p in enumerate(picks)}
    starts = sorted((s for r in reps for s in r["starts"]), key=lambda s: order[s["id"]])
    with open(out / "starts.jsonl", "w", encoding="utf-8") as fh:
        fh.write("".join(json.dumps(s) + "\n" for s in starts))
    meta.update({"replays": len(reps), "replays_same": sum(r["same"] for r in reps), "starts": len(starts),
                 "starts_by_cat": {c: sum(s["cat"] == c for s in starts) for c in ("act1-boss", "act2-elite", "act2-boss")},
                 "replay_seconds": round(sum(r["seconds"] for r in reps), 1)})
    (out / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")
    print(json.dumps({k: v for k, v in meta.items() if k != "agent_root"}, indent=1))
    return 0


def _cat_starts(out: Path, cat: str, limit: int, same_only: bool = False) -> list[dict]:
    """A category's starts in selection order. A start whose replay left the night's run before it (a later engine took another
    macro choice) is still a position the same agent reached on this engine: it is played, and counted in meta.json."""
    starts = [s for s in read_jsonl(out / "starts.jsonl") if (s["cat"] == cat or s["cat"].startswith(cat + "-"))]
    if same_only:
        starts = [s for s in starts if s.get("replay_same")]
    return starts[:limit] if limit else starts


def play_stage(out: Path, arm: str, cat: str, seeds: int, limit: int, procs: int) -> None:
    starts = _cat_starts(out, cat, limit)
    done = done_keys(read_jsonl(out / "games.jsonl"), game_key)
    tasks = [(arm, sid, j) for sid, j in todo_games(starts, arm, seeds, done)]
    guarded_run(out, f"play {arm} {cat}", tasks, _game_task, out / "games.jsonl", procs, _init_args(out))
    if arm == "a3-v1" and cat.startswith("act1"):
        _collect_leaves(out)


def _collect_leaves(out: Path) -> None:
    """Move the a3-v1 j = 0 games' first-turn leaves out of games.jsonl into m3_leaves.jsonl (once per start)."""
    have = {r["start"] for r in read_jsonl(out / "m3_leaves.jsonl")}
    with open(out / "m3_leaves.jsonl", "a", encoding="utf-8") as fh:
        for r in read_jsonl(out / "games.jsonl"):
            if r["arm"] == "a3-v1" and int(r["j"]) == 0 and r.get("_leaf") and r["start"] not in have:
                fh.write(json.dumps({"start": r["start"], "doc": r["_leaf"]}) + "\n")
                have.add(r["start"])


def m3_stage(out: Path, seeds: int, procs: int) -> None:
    _collect_leaves(out)
    leaves = [r["start"] for r in read_jsonl(out / "m3_leaves.jsonl")]
    done = {(r["type"], r["start"], int(r["j"])) for r in read_jsonl(out / "m3.jsonl")}
    tasks = [("pred", s, 0) for s in leaves if ("pred", s, 0) not in done]
    tasks += [("cont", s, j) for s in leaves for j in range(seeds) if ("cont", s, j) not in done]
    guarded_run(out, "m3", tasks, _m3_task, out / "m3.jsonl", procs, _init_args(out))


def cmd_play(a) -> int:
    play_stage(Path(a.out), a.arm, a.cat, a.seeds, a.limit, a.procs)
    return 0


def cmd_m3(a) -> int:
    m3_stage(Path(a.out), a.seeds, a.procs)
    return 0


def cmd_run(a) -> int:
    """The stages in order (arm:cat[:limit] or m3), each resumable; then the report."""
    out = Path(a.out)
    for stage in a.plan.split(","):
        part = stage.split(":")
        if part[0] == "m3":
            m3_stage(out, a.seeds, a.procs)
        else:
            play_stage(out, part[0], part[1], a.seeds, int(part[2]) if len(part) > 2 else 0, a.procs)
    _say(out, "run: done")
    return cmd_report(a)


def _pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def report(out: Path) -> dict:
    games = [r for r in read_jsonl(out / "games.jsonl")]
    starts = {s["id"]: s for s in read_jsonl(out / "starts.jsonl")}
    meta = _meta(out)
    rep: dict = {"source": meta["source"], "agent": meta["agent"], "start_point": meta.get("start_point"),
                 "replays": meta.get("replays"), "replays_same": meta.get("replays_same"), "cats": {}}
    for cat in ("act1-boss", "act2-elite", "act2-boss", "act2"):
        rows = [r for r in games if r["cat"] == cat or r["cat"].startswith(cat + "-")]
        if not rows:
            continue
        c: dict = {"arms": {}, "diffs": {}}
        for arm in ARMS:
            g = [r for r in rows if r["arm"] == arm]
            if not g:
                continue
            w = [r for r in g if r["won"]]
            rv = [r["root_value"] for r in g if r.get("root_value") is not None]
            c["arms"][arm] = {"games": len(g), "starts": len({r["start"] for r in g}), "win_rate": len(w) / len(g),
                              "end_hp_wins": sum(r["end_hp"] for r in w) / len(w) if w else None,
                              "L0": sum(r["L0"] for r in g) / len(g), "potions_used": sum(r["potions_used"] for r in g) / len(g),
                              "turns": sum(r["turns"] for r in g) / len(g), "decisions": sum(r["decisions"] for r in g) / len(g),
                              "sims": sum(r["sims"] for r in g) / len(g), "seconds": sum(r["seconds"] for r in g) / len(g),
                              "cpu_hours": sum(r["seconds"] for r in g) / 3600, "capped": sum(r["capped"] for r in g),
                              "budget_exhausted": sum(r["budget_exhausted"] for r in g),
                              "root_value": sum(rv) / len(rv) if rv else None,
                              "root_p_death": _avg([r.get("root_p_death") for r in g])}
        for arm in ("x4", "x16"):
            if arm in c["arms"] and "a3-v1" in c["arms"]:
                c["diffs"][arm] = {"won": paired_diff(rows, arm, "a3-v1", "won"), "L0": paired_diff(rows, arm, "a3-v1", "L0"),
                                   "root_value": paired_diff([r for r in rows if r.get("root_value") is not None], arm, "a3-v1", "root_value")}
        st = [starts[s] for s in {r["start"] for r in rows} if s in starts]
        c["starts_at"] = {"n": len(st), "hp": _avg([s["hp"] for s in st]), "hp_frac": _avg([s["hp"] / s["max_hp"] for s in st]),
                          "deck_size": _avg([s["deck_size"] for s in st]), "potions": _avg([s["potions"] for s in st])}
        c["by_hp_third"] = _by_hp_third(rows, starts)
        rep["cats"][cat] = c
    m3 = read_jsonl(out / "m3.jsonl")
    if m3:
        rep["m3"] = _m3_report(m3)
    return rep


def _avg(xs: list) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _by_hp_third(rows: list[dict], starts: dict) -> dict:
    fr = sorted({r["start"]: starts[r["start"]]["hp"] / starts[r["start"]]["max_hp"] for r in rows if r["start"] in starts}.items(),
                key=lambda kv: kv[1])
    if len(fr) < 3:
        return {}
    thirds = {sid: min(2, 3 * i // len(fr)) for i, (sid, _) in enumerate(fr)}
    out = {}
    for t in range(3):
        ids = {s for s, x in thirds.items() if x == t}
        hp = [v for s, v in fr if s in ids]
        out[f"t{t}"] = {"hp_frac": [round(min(hp), 3), round(max(hp), 3)],
                        **{arm: _avg([float(r["won"]) for r in rows if r["arm"] == arm and r["start"] in ids]) for arm in ARMS}}
    return out


def _m3_report(m3: list[dict]) -> dict:
    preds = {r["start"]: r for r in m3 if r["type"] == "pred"}
    cont: dict[str, list[dict]] = defaultdict(list)
    for r in m3:
        if r["type"] == "cont":
            cont[r["start"]].append(r)
    ids = sorted(s for s in preds if len(cont.get(s, [])) >= 1)
    tw = [sum(float(x["won"]) for x in cont[s]) / len(cont[s]) for s in ids]
    tl = [sum(x["L0"] for x in cont[s]) / len(cont[s]) for s in ids]
    res = {"positions": len(ids), "games": sum(len(cont[s]) for s in ids), "target_win": _avg(tw), "target_L0": _avg(tl),
           "value_head_L0": _avg([preds[s]["vh_L0"] for s in ids]), "playouts_L0": _avg([preds[s]["po_L0"] for s in ids])}
    if len(ids) >= 3:
        res["L0"] = score_predictions(tl, {"value_head": [preds[s]["vh_L0"] for s in ids],
                                           "playouts": [preds[s]["po_L0"] for s in ids]}, seed=1)
        res["win"] = score_predictions(tw, {"value_head": [preds[s]["vh_win"] for s in ids],
                                            "playouts": [preds[s]["po_win"] for s in ids]}, seed=2)
    return res


def _fmt_diff(d: dict) -> str:
    if d["diff"] is None:
        return "-"
    sp = "-" if d["se_pairs"] is None else f"{100 * d['se_pairs']:.1f}"
    ss = "-" if d["se_starts"] is None else f"{100 * d['se_starts']:.1f}"
    return f"{100 * d['diff']:+.1f} (se {sp} over {d['pairs']} pairs, {ss} over {d['starts']} starts)"


def table(rep: dict) -> str:
    lines = [f"# Fight gym M2: {rep['source']} ({rep['agent']})", "",
             f"Start point: {rep['start_point']}. Replays reproducing the arm up to the start: {rep['replays_same']} of {rep['replays']}.", ""]
    for cat, c in rep["cats"].items():
        lines += [f"## {cat}", "", "| arm | starts | games | win % | mean L0 | end HP (wins) | potions used | turns | sims/game | s/game | CPU h | root value (HP) |",
                  "|---|---|---|---|---|---|---|---|---|---|---|---|"]
        for arm, x in c["arms"].items():
            hp = "-" if x["end_hp_wins"] is None else f"{x['end_hp_wins']:.1f}"
            rv = "-" if x["root_value"] is None else f"{x['root_value']:.1f}"
            lines.append(f"| {arm} | {x['starts']} | {x['games']} | {_pct(x['win_rate'])} | {x['L0']:.3f} | {hp} | "
                         f"{x['potions_used']:.2f} | {x['turns']:.1f} | {x['sims']:.0f} | {x['seconds']:.1f} | {x['cpu_hours']:.2f} | {rv} |")
        lines.append("")
        for arm, d in c["diffs"].items():
            lines.append(f"- {arm} minus a3-v1: win {_fmt_diff(d['won'])} points; L0 {100 * (d['L0']['diff'] or 0):+.2f} x 0.01; "
                         f"first-decision root value {(d['root_value']['diff'] or 0):+.1f} HP over {d['root_value']['pairs']} pairs")
        s = c["starts_at"]
        lines.append(f"- At the start: HP {s['hp']:.1f} ({100 * s['hp_frac']:.0f}% of max), deck {s['deck_size']:.1f} cards, "
                     f"{s['potions']:.2f} potions ({s['n']} starts)")
        for t, x in c["by_hp_third"].items():
            lines.append(f"- HP third {t} ({x['hp_frac'][0]:.2f}-{x['hp_frac'][1]:.2f}): "
                         + ", ".join(f"{arm} {_pct(x[arm])}%" for arm in ARMS if x.get(arm) is not None))
        lines.append("")
    m = rep.get("m3")
    if m:
        lines += ["## M3: the horizon-0 leaf's value", "",
                  f"{m['positions']} positions, {m['games']} a3-v1 games from them; target win {_pct(m['target_win'])}%, "
                  f"mean L0 {m['target_L0']:.3f}; value head mean L0 {m['value_head_L0']:.3f}; 30 playouts mean L0 {m['playouts_L0']:.3f}.", ""]
        for what in ("L0", "win"):
            if what not in m:
                continue
            lines += [f"Against the target's {what}:", "", "| predictor | Pearson [95%] | Spearman [95%] | MSE [95%] |", "|---|---|---|---|"]
            for name in ("value_head", "playouts"):
                x = m[what][name]
                cell = [f"{x[k]:.3f} [{x[k + '_ci'][0]:.3f}, {x[k + '_ci'][1]:.3f}]" if x[k] is not None and x.get(k + "_ci") else "-"
                        for k in ("pearson", "spearman", "mse")]
                lines.append(f"| {name} | {cell[0]} | {cell[1]} | {cell[2]} |")
            b = m[what]["better_share"]
            lines.append("")
            lines.append("Share of bootstrap resamples where each is better: " + "; ".join(
                f"{k}: " + ", ".join(f"{n} {v:.2f}" for n, v in b[k].items()) for k in ("pearson", "spearman", "mse")))
            lines.append("")
    return "\n".join(lines) + "\n"


def cmd_report(a) -> int:
    out = Path(a.out)
    rep = report(out)
    (out / "report.json").write_text(json.dumps(rep, indent=1), encoding="utf-8")
    text = table(rep)
    (out / "table.md").write_text(text, encoding="utf-8")
    print(text)
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("cmd", choices=("starts", "play", "m3", "run", "report"))
    ap.add_argument("--source", default="D:/spireline/referee/lab/nightly/2026-10-27/g7-emu")
    ap.add_argument("--out", required=True)
    ap.add_argument("--agent-root", default="D:/spireline/referee/agent")
    ap.add_argument("--act1-boss", type=int, default=200)
    ap.add_argument("--act2", type=int, default=200)
    ap.add_argument("--seed", type=int, default=20261029)
    ap.add_argument("--arm", choices=tuple(ARMS), default="a3-v1")
    ap.add_argument("--cat", default="act1-boss")
    ap.add_argument("--seeds", type=int, default=10)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--plan", default="a3-v1:act1-boss,x16:act1-boss,x4:act1-boss,m3")
    ap.add_argument("--procs", type=int, default=MAX_PROCS)
    a = ap.parse_args(argv)
    return {"starts": cmd_starts, "play": cmd_play, "m3": cmd_m3, "run": cmd_run, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
