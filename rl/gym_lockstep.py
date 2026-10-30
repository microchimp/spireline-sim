"""The lockstep fight gym (gym build item P1): one process plays K fights at once and asks the network for all of them in one batch.

    python rl/fight_gym.py play --out OUT --arm a3-v1 --seeds 1 --lockstep 32 --prior cpu|cpu-stacked|server [--procs 2]
                                [--grade-wall-s 1] [--worlds reload|memory] [--games-name NAME] [--gpu-python PY]

Each fight is `fight_gym.play_steps`: its searches are `live_search.search_steps`, unchanged and in their own order, except that
wherever a search needs the prior it yields the request `(handle, legal)` instead of calling it. A process holds K fights. It
advances each until its next request (or its end), then answers every pending request in one batch, and goes round again. Nothing
is batched within one search (no virtual loss: GPU_INFERENCE.md section 9), so a fight plays exactly as it does alone; only the
network's calls are gathered across fights.

The batch is answered by one of:

- `cpu`: the agent's CPU prior (`combat_prior.LivePrior`) row by row, in process. Bit for bit the sequential gym's answers: the
  check against M2's games, and the path for the night.
- `cpu-stacked`: the same network on the CPU, one padded forward per batch (`PolicyNet.weights`). Faster per call; not bit-exact
  with a batch of one, so a fight may come out otherwise than alone (a near-tie of PUCT scores flips).
- `server`: the batched prior server (`rl/prior_server.py`, CUDA graphs) through one PRIORS frame per batch, so the card sees K
  times the processes' rows at once; the client's fallback is the CPU prior. Numerics: GPU_INFERENCE.md section 9.2.

Each fight has its own clock (`OwnClock`): the time its own steps ran plus its share of each batch it was in. Its deadline and its
searches' safety budgets read that clock, so they mean what they mean in the sequential gym; `seconds` in a row is that time, and
`wall_s` the wall time from its first step to its end.

The guards of the M2 gym, carried over to K fights a process:

- A fight past its deadline (ten times its arm's mean, between decisions) stops there, capped and lost, with its turns and
  seconds; the other fights go on.
- The parent watches each process's heartbeat (every advance of a fight beats once) and its private memory. A process whose
  heartbeat stops for `--hang-s` (a native step that never returns, as F18's) or that passes `--mem-gb` is killed by its own
  handle: the fight it was advancing gets a capped row with how far it got, and its other unfinished fights are played again from
  their start in a fresh process (a fight's outcome depends on its seeds only). A fight sent back three times is capped too.
- A process holds at most `K_MAX` fights: the emulator's pool holds 256 combats, and a suspended fight holds two (its world and a
  simulation's fork or the root's probe).

`--worlds memory` loads each start once per process and makes each seed's world a resampled fork of it in memory. It is not the
default: the load seed survives the fork (the fork reshuffles the draw pile from the order the load dealt), so seeds j >= 1 would be
other worlds than the M2 gym's, which reloads the document per seed, and the board's squares would no longer pair. A reload costs
about 3 ms a game (0.05% of an a3-v1 boss game), so `reload` keeps the pairing for nothing.

`--grade-wall-s S` grades each decision of the played fight with the exact solver: one expected solve (`native.solve_expected`;
the only solver output a grade may read) at horizon 0 under an S-second wall, before the move is stepped. The grade records the
solve's value, the value of the move the search chose, whether horizon 0 finished and whether the value is exact (the fight ends
within the turn: a last turn). It never changes the move played.

Lab code: nothing here changes the engine, the search or the solver. No QUIET.lock line is taken (a two-core job); the job waits
while the night is on or due, while main.lock exists, or while another live QUIET.lock line stands (`fight_gym.blocked`). The
server path takes GPU_OK's first line only while it runs, and only when the file is empty or absent.
"""

from __future__ import annotations

import argparse
import collections
import ctypes
import importlib
import itertools
import json
import multiprocessing as mp
import multiprocessing.connection
import os
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fight_gym as fg  # noqa: E402
import live_search  # noqa: E402 -- the search's steps; loads the emulator library
from sts2_gym import native  # noqa: E402

TAG = "gym-gpu"
K_MAX = 100  # fights a process holds: two combats each, inside the emulator's pool of 256 with room for the rest
HANG_S = 180.0  # no heartbeat this long while a process holds fights: a step that never returns
MEM_LIMIT_GB = 3.0
MAX_TRIES = 3  # a fight sent back to a fresh process this many times is capped
BASES = 8  # loaded starts a process keeps for `--worlds memory`
PRIORS = ("cpu", "cpu-stacked", "server", "uniform")  # uniform: the tests' stand-in, never a measurement
GPU_OK = fg.REF / "GPU_OK"
GPU_OFF = fg.REF / "LAB" / "GPU_OFF"
GPU_PYTHON = os.environ.get("SPIRELINE_GPU_PYTHON", "D:/spireline/venv-gpu/Scripts/python.exe")


# ---------------------------------------------------------------------------------------------------------- the pieces


class OwnClock:
    """A fight's own clock: the time its steps ran (between `start` and `stop`) plus the batch time charged to it."""

    __slots__ = ("acc", "since")

    def __init__(self) -> None:
        self.acc = 0.0
        self.since: float | None = None

    def __call__(self) -> float:
        return self.acc + (time.perf_counter() - self.since if self.since is not None else 0.0)

    def start(self) -> None:
        self.since = time.perf_counter()

    def stop(self) -> None:
        if self.since is not None:
            self.acc += time.perf_counter() - self.since
            self.since = None


class CpuExact:
    """The agent's CPU prior, row by row: the sequential gym's answers, bit for bit."""

    name = "cpu"

    def __init__(self, prior_fn) -> None:
        self.prior_fn = prior_fn

    def __call__(self, requests: list) -> list:
        out = []
        for handle, legal in requests:
            try:
                out.append(self.prior_fn(handle, legal))
            except Exception as error:  # noqa: BLE001 -- raised inside that fight's search, where it asked
                out.append(error)
        return out

    def stats(self) -> dict:
        return {}


class CpuStacked:
    """The agent's network on the CPU, one padded forward per batch."""

    name = "cpu-stacked"

    def __init__(self, prior_fn) -> None:
        self.policy, self.probe = prior_fn.policy, prior_fn.probe

    def __call__(self, requests: list) -> list:
        import combat_prior  # noqa: PLC0415
        from prior_client import encode_position  # noqa: PLC0415

        tokens = [encode_position(handle, self.probe) for handle, _ in requests]
        return combat_prior.network(self.policy).weights(tokens, [list(legal) for _, legal in requests])

    def stats(self) -> dict:
        return {}


class ServerBatch:
    """The batched prior server: one PRIORS frame per batch (`prior_client.RemotePrior.batch`), the CPU prior on failure."""

    name = "server"

    def __init__(self, remote) -> None:
        self.remote = remote

    def __call__(self, requests: list) -> list:
        return self.remote.batch(requests)

    def stats(self) -> dict:
        import prior_client  # noqa: PLC0415

        return prior_client.stats()


class Uniform:
    """Equal weight per legal action. For the tests' stand-in fights only."""

    name = "uniform"

    def __call__(self, requests: list) -> list:
        import numpy as np  # noqa: PLC0415

        return [np.full(len(legal), 1.0 / max(1, len(legal))) for _, legal in requests]

    def stats(self) -> dict:
        return {}


def make_evaluator(prior: str, prior_fn, server: dict | None = None):
    if prior == "uniform":
        return Uniform()
    if prior_fn is None:
        return CpuExact(None)  # an agent without a prior asks for nothing
    if prior == "cpu":
        return CpuExact(prior_fn)
    if prior == "cpu-stacked":
        return CpuStacked(prior_fn)
    if prior == "server":
        import prior_client  # noqa: PLC0415

        return ServerBatch(prior_client.RemotePrior(policy=prior_fn.policy, probe=prior_fn.probe, endpoint=server["endpoint"],
                                                    authkey=server["authkey"], policy_sha256=server["policy_sha256"]))
    raise ValueError(f"unknown prior path {prior!r}; one of {', '.join(PRIORS)}")


class Grader:
    """The exact solver's grade of a played decision: one expected solve at a fixed horizon (0) under a wall, on a fresh solver.

    The row: the solve's status, whether the value is exact and whether horizon 0 finished, the value, the value of the move the
    search chose (`q_played`, mapped onto the solver's keys as coverage.py's regret does) and the solver's best move. It reads the
    expected solve only, and only the position: the move played is the search's, whatever the grade says."""

    def __init__(self, wall_s: float, horizon: int = 0) -> None:
        self.wall_s, self.horizon = float(wall_s), int(horizon)
        self.seconds = 0.0
        self.calls = 0

    def __call__(self, world, k: int, best) -> dict:
        from tools.solver import coverage  # noqa: PLC0415

        t0 = time.perf_counter()
        opts = native.SolveOptions(time_budget_ms=max(1, round(self.wall_s * 1000)), max_horizon=self.horizon)
        out: dict = {"k": k, "move": [best.action, best.target, best.kind]}
        try:
            solver = native.solver_create(world.handle, opts)
            try:
                r = native.solve_expected(solver, world.handle, opts)
                horizons = native.solver_stats(solver).get("horizons") or []
            finally:
                native.solver_destroy(solver)
            key, how = coverage.map_played(r.q, best.action, best.target, None)
            if key is None and r.q:
                try:
                    key, how = coverage.map_played(r.q, best.action, best.target, native.get_state_json(world.handle))
                except Exception:  # noqa: BLE001, S110 -- a position stopped mid-step has no document: left unmapped
                    pass
            out.update(status=r.status, exact=bool(r.exact), horizon=r.horizon,
                       h0=any(h.get("horizon") == 0 and (h.get("end") == "done" or h.get("exact")) for h in horizons),
                       value=r.value, q_played=r.q.get(key) if key is not None else None, map=how,
                       solver_best=[r.best_action, r.best_target], exact_mass=r.exact_mass)
        except Exception as error:  # noqa: BLE001 -- a grade that fails is recorded, never a stopped game
            out["error"] = f"{type(error).__name__}: {error}"[:200]
        dt = time.perf_counter() - t0
        out["s"] = round(dt, 4)
        self.seconds += dt
        self.calls += 1
        return out


def grade_summary(grades: list[dict]) -> dict:
    """Per game: decisions graded, horizon 0 finished, exact, and the summed regret (value - q_played) where exact."""
    ok = [g for g in grades if "error" not in g and g.get("status") != native.SOLVE_REFUSED]
    exact = [g for g in ok if g.get("exact") and g.get("value") is not None and g.get("q_played") is not None]
    return {"graded": len(grades), "h0": sum(bool(g.get("h0")) for g in ok), "exact": len(exact),
            "regret_exact": round(sum(g["value"] - g["q_played"] for g in exact), 6), "seconds": round(sum(g["s"] for g in grades), 3)}


# ---------------------------------------------------------------------------------------------------------- the process


def _world(start: dict, j: int, worlds: str, bases: collections.OrderedDict):
    """Game j's world: the document reloaded with j's seeds (the M2 gym's `_fresh_world`), or a fork of the start loaded once."""
    if worlds == "reload":
        return fg._fresh_world(start["doc"], start["id"], j, "g")  # noqa: SLF001
    import combat_mcts  # noqa: PLC0415

    base = bases.pop(start["id"], None)
    if base is None:
        base = combat_mcts.Combat(native.load_combat_snapshot(start["doc"], fg.game_seed(start["id"], 0, "gload")))
    bases[start["id"]] = base
    while len(bases) > BASES:
        bases.popitem(last=False)[1].close()
    world = base.fork(fg.game_seed(start["id"], j, "gfork"))
    world.encounter_id = native.encounter_id(world.handle)
    return world


def _game_steps(task: tuple, clock: OwnClock, game, ctx):
    """One gym game as steps: `fight_gym._game_task` with its prior calls yielded."""
    arm, start_id, j = task[:3]
    start = ctx.starts[start_id]
    world = _world(start, j, ctx.worlds, ctx.bases)
    try:
        row = yield from fg.play_steps(world, start, j, fg.arm_settings(arm)["cfg"], ctx.extra, "g",
                                       capture=arm == "a3-v1" and j == 0 and start["cat"] == "act1-boss",
                                       deadline=fg.task_deadline(task), clock=clock, progress=game.note, grader=ctx.grader)
    finally:
        world.close()
    if "grades" in row:
        row["graded"] = grade_summary(row["grades"])
    return {"_task": [arm, start_id, j], "arm": arm, "start": start_id, "j": j, "cat": start["cat"], **row}


def _resolve(name: str | None):
    if not name:
        return None
    module, _, attr = name.partition(":")
    return getattr(importlib.import_module(module), attr)


class _Context:
    def __init__(self, opts: dict) -> None:
        self.k = int(opts["k"])
        self.worlds = opts.get("worlds", "reload")
        self.bases: collections.OrderedDict = collections.OrderedDict()
        for p in opts.get("paths", ()):
            if p not in sys.path:
                sys.path.insert(0, p)
        self.game_steps = _resolve(opts.get("game_steps")) or _game_steps
        agent = fg._W.get("agent")  # noqa: SLF001
        self.extra = agent.search_extra() if agent is not None else {}
        self.starts = fg._starts_by_id() if self.game_steps is _game_steps else {}  # noqa: SLF001
        self.grader = Grader(opts["grade_wall_s"]) if opts.get("grade_wall_s") else None
        self.evaluator = make_evaluator(opts["prior"], self.extra.get("prior_fn"), opts.get("server"))
        self.label = {"lockstep": self.k, "prior_path": self.evaluator.name, "worlds": self.worlds}


class _Game:
    __slots__ = ("serial", "task", "ctx", "clock", "gen", "request", "answer", "row", "prog", "prog_new", "t_first")

    def __init__(self, serial: int, task: tuple, ctx: _Context) -> None:
        self.serial, self.task, self.ctx = serial, task, ctx
        self.clock = OwnClock()
        self.gen = self.request = self.answer = self.row = None
        self.prog: dict = {}
        self.prog_new = False
        self.t_first = 0.0

    def note(self, d: dict) -> None:
        self.prog = d
        self.prog_new = True

    def advance(self) -> None:
        """Run the game to its next request or its end."""
        self.clock.start()
        try:
            if self.gen is None:
                self.t_first = time.perf_counter()
                self.gen = self.ctx.game_steps(self.task, self.clock, self, self.ctx)
                self.request = next(self.gen)
            elif isinstance(self.answer, BaseException):
                self.request = self.gen.throw(self.answer)
            else:
                self.request = self.gen.send(self.answer)
        except StopIteration as done:
            self.request, self.row = None, done.value
        except Exception as error:  # noqa: BLE001 -- the game's capped row, as the M2 gym records a raising task
            self.request = None
            self.row = fg.capped_row(self.task, f"error: {type(error).__name__}: {error}"[:300], self.clock(), self.prog)
        finally:
            self.answer = None
            self.clock.stop()

    def final_row(self) -> dict:
        return {**self.row, **self.ctx.label, "wall_s": round(time.perf_counter() - self.t_first, 2)}


def _child_main(conn, shared, init_args: tuple, opts: dict) -> None:
    """A lockstep process: up to K games at once off its pipe; rows, progress and at the end its stats go back the same way.

    shared[0] is the serial of the game being advanced (-1 between games, -2 starting up), shared[1] the heartbeat: one beat
    per advance. The parent reads both without a message, so a step that never returns is seen and its game named."""
    try:
        if init_args:
            fg._init_worker(*init_args)  # noqa: SLF001 -- BelowNormal, the engine, the agent, the starts
        else:
            from tools.solver import coverage  # noqa: PLC0415

            coverage._below_normal()  # noqa: SLF001
        ctx = _Context(opts)
    except Exception as error:  # noqa: BLE001 -- the parent stops the job: a process that cannot start never will
        conn.send(("fatal", f"{type(error).__name__}: {error}"[:300]))
        return
    games: list[_Game] = []
    queue: collections.deque = collections.deque()
    closing = False
    hist: collections.Counter = collections.Counter()
    t_net = t_games = t_idle = 0.0
    rounds = requests = done = 0
    t_start = time.perf_counter()
    shared[0] = -1
    while True:
        while conn.poll():
            msg = conn.recv()
            if msg is None:
                closing = True
            else:
                queue.append(msg)
        while len(games) < ctx.k and queue:
            serial, task = queue.popleft()
            games.append(_Game(serial, task, ctx))
        if not games:
            if closing:
                break
            t0 = time.perf_counter()
            msg = conn.recv()
            t_idle += time.perf_counter() - t0
            if msg is None:
                closing = True
            else:
                queue.append(msg)
            continue
        t0 = time.perf_counter()
        for g in games:
            if g.request is None and g.row is None:
                shared[0] = g.serial
                g.advance()
                shared[1] += 1
                if g.row is not None:  # at once: a later game of the round may never come back
                    conn.send(("row", g.serial, g.final_row()))
                    done += 1
                elif g.prog_new:
                    conn.send(("prog", g.serial, g.prog))
                    g.prog_new = False
        shared[0] = -1
        t_games += time.perf_counter() - t0
        games = [g for g in games if g.row is None]
        pending = [g for g in games if g.request is not None]
        if pending:
            t0 = time.perf_counter()
            try:
                answers = ctx.evaluator([g.request for g in pending])
            except Exception as error:  # noqa: BLE001 -- raised inside every waiting search
                answers = [error] * len(pending)
            dt = time.perf_counter() - t0
            t_net += dt
            for g, answer in zip(pending, answers):
                g.request, g.answer = None, answer
                g.clock.acc += dt / len(pending)
            hist[len(pending)] += 1
            rounds += 1
            requests += len(pending)
    total = time.perf_counter() - t_start
    conn.send(("stats", {"games": done, "rounds": rounds, "requests": requests, "batch_hist": {str(k): v for k, v in sorted(hist.items())},
                         "wall_s": round(total, 3), "net_s": round(t_net, 3), "games_s": round(t_games, 3), "idle_s": round(t_idle, 3),
                         "grade_s": round(ctx.grader.seconds, 3) if ctx.grader else 0.0, "cpu_s": round(time.process_time(), 3),
                         "evaluator": ctx.evaluator.name, "evaluator_stats": ctx.evaluator.stats()}))


# ---------------------------------------------------------------------------------------------------------- the parent


def run_lockstep(tasks: list, target: Path, procs: int, k: int, init_args: tuple, opts: dict, *, hang_s: float = HANG_S,
                 mem_limit: float | None = MEM_LIMIT_GB * 2**30, stop=None, poll_s: float = 0.5, log=None) -> tuple[str | None, list]:
    """Play `tasks` in `procs` lockstep processes of `k` games each, appending each row to `target` as it comes.

    Returns (why it stopped or None, the processes' stats). Never waits forever: see the module docstring for the guards."""
    if not 1 <= k <= K_MAX:
        raise ValueError(f"K is 1..{K_MAX}: the emulator's pool holds 256 combats and a waiting game holds two")
    ctx = mp.get_context("spawn")
    left = collections.deque(tasks)
    serials = itertools.count()
    tries: collections.Counter = collections.Counter()
    stats: list[dict] = []
    hold = k + max(2, k // 2)  # games a process holds: its K and a few queued
    say = log or (lambda _m: None)

    def spawn() -> dict:
        mine, theirs = ctx.Pipe()
        shared = ctx.RawArray(ctypes.c_longlong, 2)
        shared[0] = -2
        p = ctx.Process(target=_child_main, args=(theirs, shared, init_args, {**opts, "k": k}), daemon=True)
        p.start()
        theirs.close()
        return {"p": p, "c": mine, "sh": shared, "held": {}, "sent": {}, "prog": {}, "beat": 0, "beat_t": time.perf_counter()}

    def kill(s: dict) -> None:
        if s["p"].is_alive():
            s["p"].kill()  # this job's own child, by its handle
        s["p"].join(10)
        s["c"].close()

    slots = [spawn() for _ in range(max(1, min(fg.MAX_PROCS, procs, max(1, len(left)))))]
    stopped, asked = None, time.time()
    with open(target, "a", encoding="utf-8") as fh:

        def write(row: dict) -> None:
            fh.write(json.dumps(row) + "\n")
            fh.flush()

        def feed() -> None:
            """Hand out queued games one at a time to the process holding fewest, up to `hold` each: the processes stay even."""
            while left:
                room = [s for s in slots if len(s["held"]) < hold]
                if not room:
                    return
                s = min(room, key=lambda x: len(x["held"]))
                task = left.popleft()
                n = next(serials)
                s["held"][n], s["sent"][n] = task, time.perf_counter()
                s["c"].send((n, task))

        def take(s: dict, msg: tuple) -> None:
            if msg[0] == "row":
                task = s["held"].pop(msg[1], None)
                s["sent"].pop(msg[1], None)
                s["prog"].pop(msg[1], None)
                if task is not None:
                    write(msg[2])
            elif msg[0] == "prog":
                s["prog"][msg[1]] = msg[2]
            elif msg[0] == "stats":
                stats.append(msg[1])
            elif msg[0] == "fatal":
                raise RuntimeError(f"a lockstep process could not start: {msg[1]}")

        def drain(s: dict) -> None:
            """What a process sent before it went (a row it finished just before a hang) is still in its pipe."""
            try:
                while s["c"].poll():
                    take(s, s["c"].recv())
            except (EOFError, OSError):
                pass

        def lose(s: dict, reason: str) -> None:
            """The process is gone: its current game is capped, the rest go back to the queue (capped after MAX_TRIES)."""
            drain(s)
            cur, now, back = int(s["sh"][0]), time.perf_counter(), []
            for n, task in sorted(s["held"].items()):
                if n == cur:
                    write(fg.capped_row(task, reason, now - s["sent"][n], s["prog"].get(n, {})))
                    say(f"{reason}: {task[1]} j={task[2]} capped")
                    continue
                tid = json.dumps(fg._task_id(task))  # noqa: SLF001
                tries[tid] += 1
                if tries[tid] >= MAX_TRIES:
                    write(fg.capped_row(task, f"sent back {MAX_TRIES} times ({reason})", now - s["sent"][n], s["prog"].get(n, {})))
                else:
                    back.append(task)
            left.extendleft(reversed(back))
            if back:
                say(f"{reason}: {len(back)} unfinished games go to a fresh process")
            s["held"].clear()

        try:
            feed()
            while left or any(s["held"] for s in slots):
                busy = [s["c"] for s in slots if s["held"]]
                ready = mp.connection.wait(busy, timeout=poll_s) if busy else []
                for i, s in enumerate(slots):
                    if s["c"] not in ready:
                        continue
                    try:
                        while s["c"].poll():
                            take(s, s["c"].recv())
                    except (EOFError, OSError):
                        lose(s, "worker died")
                        kill(s)
                        slots[i] = spawn()
                feed()
                now = time.perf_counter()
                for i, s in enumerate(slots):
                    beat = int(s["sh"][1])
                    if beat != s["beat"] or not s["held"]:
                        s["beat"], s["beat_t"] = beat, now
                        if not s["held"]:
                            continue
                    reason = None
                    if now - s["beat_t"] > hang_s:
                        reason = f"no step for {hang_s:.0f} s: hung (killed)"
                    elif mem_limit is not None and (fg.private_bytes(s["p"].pid) or 0) > mem_limit:
                        reason = f"memory over {mem_limit / 2**30:.1f} GB (killed)"
                    if reason:
                        s["p"].kill()  # this job's own child, by its handle; its pipe is read to the end before it closes
                        s["p"].join(10)
                        lose(s, reason)
                        kill(s)
                        slots[i] = spawn()
                        feed()
                if stop is not None and time.time() - asked > 10:
                    asked = time.time()
                    stopped = stop()
                    if stopped:
                        break
            if not stopped:  # every game has its row: collect each process's stats
                for s in slots:
                    try:
                        s["c"].send(None)
                    except OSError:
                        continue
                end = time.perf_counter() + 60
                for s in slots:
                    while time.perf_counter() < end:
                        try:
                            if not s["c"].poll(max(0.0, end - time.perf_counter())):
                                break
                            msg = s["c"].recv()
                        except (EOFError, OSError):
                            break
                        if msg[0] == "stats":
                            stats.append(msg[1])
                            break
        finally:
            for s in slots:
                kill(s)
    return stopped, stats


# ---------------------------------------------------------------------------------------------------------- the job


def _gpu_lines() -> list[str]:
    try:
        return [x for x in GPU_OK.read_text(encoding="utf-8").splitlines() if x.strip()]
    except FileNotFoundError:
        return []


def take_gpu(tag: str = TAG) -> bool:
    """GPU_OK's first line for `tag`, only when the file is empty or absent (or `tag` holds it already)."""
    lines = _gpu_lines()
    if lines and lines[0] != tag:
        return False
    GPU_OK.write_text("".join(x + "\n" for x in [tag] + [x for x in lines if x != tag]), encoding="utf-8", newline="\n")
    return True


def drop_gpu(tag: str = TAG) -> None:
    lines = [x for x in _gpu_lines() if x != tag]
    if lines:
        GPU_OK.write_text("".join(x + "\n" for x in lines), encoding="utf-8", newline="\n")
    else:
        GPU_OK.unlink(missing_ok=True)


class GpuSampler:
    """nvidia-smi's utilisation once a second while the server path runs: the card's busy share."""

    def __init__(self) -> None:
        self.samples: list[float] = []
        self.proc = None
        exe = shutil.which("nvidia-smi")
        if exe:
            self.proc = subprocess.Popen([exe, "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits", "-l", "1"],  # noqa: S603
                                         stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True)
            threading.Thread(target=self._read, daemon=True).start()

    def _read(self) -> None:
        for line in self.proc.stdout:
            try:
                self.samples.append(float(line.strip()))
            except ValueError:
                continue

    def stop(self) -> dict:
        if self.proc is not None:
            self.proc.terminate()
            try:
                self.proc.wait(5)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        s = self.samples
        return {"n": len(s), "mean_pct": round(sum(s) / len(s), 1) if s else None, "busy_share": round(sum(x > 0 for x in s) / len(s), 3) if s else None}


def _champion(out: Path):
    import combat_prior  # noqa: PLC0415
    import gen_loop  # noqa: PLC0415

    meta = fg._meta(out)  # noqa: SLF001
    agent = gen_loop.load_agent(Path(meta["agent_root"]) / meta["agent"])
    return combat_prior.find(agent.prior_dir) if agent.prior_dir else None


def play(out: Path, arm: str, cat: str, seeds: int, limit: int, procs: int, k: int, prior: str, *, grade_wall_s: float = 0.0,
         worlds: str = "reload", games_name: str = "games.jsonl", gpu_python: str | None = None, hang_s: float = HANG_S,
         mem_gb: float = MEM_LIMIT_GB) -> dict:
    """Play an arm's games of a category in lockstep (resumable: done games are skipped), then write `<games>.stats.json`."""
    from tools.solver import coverage  # noqa: PLC0415

    coverage._below_normal()  # noqa: SLF001 -- the parent too
    out = Path(out)
    target = out / games_name
    starts = fg._cat_starts(out, cat, limit)  # noqa: SLF001
    done = fg.done_keys(fg.read_jsonl(target), fg.game_key)
    cats = {s["id"]: s["cat"] for s in starts}
    tasks = [(arm, sid, j, cats[sid]) for sid, j in fg.todo_games(starts, arm, seeds, done)]
    opts = {"prior": prior, "grade_wall_s": grade_wall_s, "worlds": worlds}
    what = f"lockstep {arm} {cat} K={k} {prior}{' graded' if grade_wall_s else ''} -> {games_name}"
    server = sampler = None
    took_gpu = False
    stats: list = []
    computing = 0.0
    played = 0
    try:
        if prior == "server" and tasks:
            from prior_launch import start_prior_server  # noqa: PLC0415

            while not take_gpu():
                fg._say(out, f"{what}: waiting for GPU_OK ({_gpu_lines()[:1]})")  # noqa: SLF001
                time.sleep(60)
            took_gpu = True
            champion = _champion(out)
            server = start_prior_server(gpu_python or GPU_PYTHON, champion.policy, lib_path=ROOT / "out", kill_file=GPU_OFF,
                                        stats_path=out / (games_name + ".server.json"))
            if server is None:
                raise RuntimeError("the prior server did not start")
            opts["server"] = {"endpoint": server.endpoint, "authkey": server.authkey, "policy_sha256": champion.policy_sha256}
            sampler = GpuSampler()
        left = tasks
        while left:
            why = fg.blocked()
            if why:
                fg._say(out, f"{what}: waiting ({why}); {len(left)} left")  # noqa: SLF001
                while fg.blocked():
                    time.sleep(60)
            fg._say(out, f"{what}: computing; {len(left)} left")  # noqa: SLF001
            t0 = time.perf_counter()
            stopped, got = run_lockstep(left, target, procs, k, fg._init_args(out), opts, hang_s=hang_s,  # noqa: SLF001
                                        mem_limit=mem_gb * 2**30, stop=fg.blocked, log=lambda m: fg._say(out, f"{what}: {m}"))  # noqa: SLF001
            computing += time.perf_counter() - t0
            stats += got
            gone = {json.dumps(r.get("_task")) for r in fg.read_jsonl(target)}
            before = len(left)
            left = [t for t in left if json.dumps(fg._task_id(t)) not in gone]  # noqa: SLF001
            played += before - len(left)
            if stopped:
                fg._say(out, f"{what}: stopped ({stopped}); {len(left)} left")  # noqa: SLF001
    finally:
        gpu = sampler.stop() if sampler is not None else None
        if server is not None:
            server.stop()
        if took_gpu:
            drop_gpu()
    summary = {"arm": arm, "cat": cat, "k": k, "prior": prior, "procs": procs, "worlds": worlds, "grade_wall_s": grade_wall_s,
               "games": played, "computing_s": round(computing, 1),
               "fights_per_hour": round(3600 * played / computing, 1) if computing > 0 and played else None,
               "processes": stats, "gpu": gpu}
    if server is not None:
        summary["server"] = server.read_stats()
    capped = sum(1 for r in fg.read_jsonl(target) if r.get("cap_reason") and r.get("arm") == arm)
    fg._say(out, f"{what}: done, {played} games in {computing:.0f} s ({capped} rows capped)")  # noqa: SLF001
    (out / (games_name + ".stats.json")).write_text(json.dumps(summary, indent=1), encoding="utf-8")
    return summary


def cmd_play(a) -> int:
    summary = play(Path(a.out), a.arm, a.cat, a.seeds, a.limit, a.procs, a.lockstep, a.prior, grade_wall_s=a.grade_wall_s,
                   worlds=a.worlds, games_name=a.games_name, gpu_python=a.gpu_python, hang_s=a.hang_s, mem_gb=a.mem_gb)
    print(json.dumps({k: v for k, v in summary.items() if k not in ("processes", "server")}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True)
    ap.add_argument("--arm", choices=tuple(fg.ARMS), default="a3-v1")
    ap.add_argument("--cat", default="act1-boss")
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--procs", type=int, default=2)
    fg.add_lockstep_arguments(ap)
    a = ap.parse_args(argv)
    a.lockstep = a.lockstep or 1
    return cmd_play(a)


if __name__ == "__main__":
    raise SystemExit(main())
