"""The always-on fight-gym loop (D40, gym build item L1): generation, a trainer on the card, a promotion gate, the board.

    python rl/gym_loop.py starts  --root ROOT [--runs-per-night 50] [--procs 6]     the training start set (resumable)
    python rl/gym_loop.py weights --root ROOT [--procs 6]                           the start mix's kind weights (a3-v1 pass)
    python rl/gym_loop.py run     --root ROOT [--player-config a3-v1] [--procs 6]   the daemon (detached: see GYM_LOOP.md)
    python rl/gym_loop.py gate    --root ROOT --candidate DIR                      one promotion gate, by hand
    python rl/gym_loop.py stop    --root ROOT                                       stop the daemon (its STOP file, then its PID)
    python rl/gym_loop.py status  --root ROOT

The daemon does, in turn, while the lab's locks allow it (`why_blocked`):

1. **Generation.** A segment of fights (about `--segment-min` minutes of six processes) from the start mix: every fight kind of
   Acts 1 and 2 from every night's emulator arm, sampled by kind in proportion to where the runs lose HP (`kind_weights`), then
   uniformly within the kind. The player is the newest checkpoint (the incumbent before the first one) under the teacher search
   (`--teacher`, x16), played by P1's lockstep processes (`gym_lockstep.run_lockstep`) with the prior on the card when GPU_OK allows,
   else on the CPU. Every decision is recorded (`gen_game_steps`): the position's tokens as the prior reads them, the root's visit
   distribution and mean values (`gen_loop.root_targets`, the self-play layout), the legal mask, and the exact solver's grade where
   it finishes (`gym_lockstep.Grader`, expected solve only). Each fight's samples go to one file; after the segment they are
   compacted into one shard of the replay buffer, with the fight's outcome as the value label (end HP, 0 for a death; a capped fight
   has no value label), the player version that made it, its kind, its L0 and its split (by run seed).
2. **Training** runs beside it in its own process on the card (`rl/gym_loop_train.py`, the GPU venv): over the buffer's newest
   window, from the incumbent's weights, a checkpoint every 30 minutes, at most `--max-epochs` epochs over unchanged data (then it
   idles and logs the idle time).
3. **The gate.** A checkpoint not yet scored is played on the fight board's 200 Act-1 boss starts with the board's seeds, under the
   player config (`--player-config`: a3-v1, or the judge-equipped config once it lands), paired with the incumbent on the same games
   (the incumbent's games are played once per incumbent, engine and config, and kept). The rule (`PROMOTION_RULE`) was fixed here
   before the first checkpoint; `decide` applies it. A promoted checkpoint becomes "Fight player, version N": its games get a
   fight-board row, committed to Spireline main (`board_commit`).

Between steps the daemon drops its QUIET.lock line and waits `--yield-s` so a waiting gate can take the lock; it never holds the
line through the night (03:10-03:45Z and while LAB/nightly/active exists), and starts nothing while main.lock, LAB/PAUSE or LAB/HALT
exist or another live QUIET.lock line stands. A running step is stopped at once by the night, PAUSE, HALT, its STOP file or another
QUIET.lock line; main.lock lets it finish (as the lab's guards treat a running job). Every process is BelowNormal and in the daemon's
Job Object, so nothing it started outlives it; it is stopped by its STOP file or by its own PID. Everything is resumable: the buffer,
the checkpoints, the gate's games and the promotion state are on disk, and a restart carries on from them.

Lab code: nothing here changes the engine, the search or the solver. The human corpus's exam fifth (D38, H0) is never read: the
start set comes from the AI's own nightly runs only.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fight_gym as fg  # noqa: E402

# Nothing here loads the emulator library at import (gym_lockstep's rule): a spawned process selects its engine first.

TAG = "gym-loop"  # QUIET.lock and GPU_OK line of the daemon
TRAIN_TAG = "gym-loop-train"  # GPU_OK line of the trainer
MY_TAGS = (TAG, TRAIN_TAG)
REF = fg.REF
LAB = REF / "LAB"
PAUSE, HALT = LAB / "PAUSE", LAB / "HALT"
GPU_OK = REF / "GPU_OK"
GPU_OFF = LAB / "GPU_OFF"
GPU_PYTHON = os.environ.get("SPIRELINE_GPU_PYTHON", "D:/spireline/venv-gpu/Scripts/python.exe")
DEFAULT_ROOT = REF / "gym-loop"
STATUS_DIR = Path(os.environ.get("GYM_LOOP_STATUS_DIR", "D:/spireline/logs/gym-loop"))
BOARD_DIR = REF / "lab" / "results" / "fight-gym" / "2026-10-29-m2"  # the fight board's starts and its three standing arms
INCUMBENT0 = REF / "agent" / "gen7" / "candidate-r1"  # M2's agent: version 0, the board's "Usual thinking"
NIGHTS = ("2026-10-26/g5-emu", "2026-10-27/g7-emu", "2026-10-29/g7-emu", "2026-10-30/g9-emu")
ENGINE = "core2"
NS_GEN, NS_WEIGHTS = "L", "W"  # generation's and the weights pass's seed namespaces; the gym's (and the gate's) is "g"
KINDS = ("act1-hallway", "act1-elite", "act1-boss", "act2-hallway", "act2-elite", "act2-boss")
NODE_KIND = {"M": "hallway", "E": "elite", "B": "boss"}
NODE_TYPE = {"hallway": 1, "elite": fg.NODE_ELITE, "boss": fg.NODE_BOSS}  # run_train.vocab's node types
LAST_ACT2_INDEX = 31  # the run row's nodes: index i is floor i + 2; Acts 1 and 2 are indices 0..31 (floor 33, the Act-2 boss)
VAL_SHARE = 0.1  # run seeds whose fights are the trainer's held-out set (split by run seed, never by start: no deck in both)
MAX_ACTIONS = 32
QUIET_STALE_H = 2.0
NIGHT_WINDOW = fg.NIGHT_WINDOW
# Own seconds per x16 game by kind before any is measured (M2 and P1's numbers; a hallway fight is a short elite's).
MEAN_X16_S = {"act1-hallway": 12.0, "act1-elite": 25.0, "act1-boss": 74.0, "act2-hallway": 14.0, "act2-elite": 16.0,
              "act2-boss": 95.0}

# The promotion rule, fixed before the first checkpoint (2026-10-30). Both arms play the same games: the board's 200 Act-1 boss
# starts, game j of each with the gym's seeds (namespace "g"), the same search config; only the network differs. A capped game is
# lost. Stage 1 plays j = 0..stage1_seeds-1 and stops (no promotion) unless the candidate's win rate is at least `futility_points`
# above the incumbent's. Stage 2 plays j = 0..seeds-1 (the board's 2,000 games) and promotes only when the exact one-sided McNemar
# test on the discordant pairs (candidate-only wins against incumbent-only wins) gives p < alpha and the paired mean L0 difference
# is not negative. Nothing tunes on these games: the trainer never sees the board's starts or their runs.
PROMOTION_RULE = {"name": "gym-loop-promote-v1", "fixed": "2026-10-30, before the first checkpoint", "starts": 200,
                  "cat": "act1-boss", "stage1_seeds": 2, "futility_points": 1.0, "seeds": 10, "alpha": 0.05,
                  "l0_not_worse": True, "capped": "lost", "ns": "g"}

_HOME = re.compile(r"(?i)[a-z]:[\\/]+users[\\/]+[^\\/\s'\"]+")


def clean(text: str) -> str:
    """No user paths in logs or results: a Windows home folder becomes `~`."""
    return _HOME.sub("~", str(text))


def utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def stamp() -> str:
    return utc().strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(p: Path, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(p: Path, obj) -> None:
    """Atomic: written beside, then renamed over."""
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8", newline="\n")
    os.replace(tmp, p)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_val_seed(run_seed: str) -> bool:
    """The trainer's held-out split, by run seed: a fixed tenth of seeds."""
    return int(hashlib.sha256(f"gym-loop-val:{run_seed}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < VAL_SHARE


# ---------------------------------------------------------------------------------------------------------- the locks


def _lines(p: Path) -> list[str]:
    try:
        return [x.strip() for x in p.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]
    except FileNotFoundError:
        return []
    except OSError:
        return []


def other_quiet(lines: list[str] | None = None, *, allow: tuple[str, ...] = ()) -> str | None:
    """The tag of another live QUIET.lock line (not ours, not w3-blocks, not older than 2 h, not in `allow`), or None."""
    for x in _lines(fg.QUIET) if lines is None else lines:
        tag = x.split()[0]
        if tag in MY_TAGS or tag.startswith(TAG + "-") or tag == "w3-blocks" or tag in allow:
            continue
        if fg._age_h(x) < QUIET_STALE_H:  # noqa: SLF001
            return tag
    return None


def in_night_window(now: dt.datetime | None = None) -> bool:
    t = (now or utc()).time()
    return NIGHT_WINDOW[0] <= t < NIGHT_WINDOW[1]


def night_on() -> bool:
    return fg.ACTIVE.exists() or in_night_window() or any(x.startswith("lab-nightly") for x in _lines(fg.QUIET))


def why_blocked(root: Path, *, starting: bool) -> str | None:
    """Why the daemon must not run heavy work now (None: it may). `starting`: about to start a step (main.lock stops a start only:
    a running step carries on through a landing, as the lab's guards let a running job)."""
    if (root / "STOP").exists():
        return "stop"
    if HALT.exists():
        return "halt"
    if fg.ACTIVE.exists():
        return "nightly-active"
    if in_night_window():
        return "night-window"
    if PAUSE.exists():
        return "pause"
    if starting and fg.MAIN_LOCK.exists():
        return "main.lock"
    tag = other_quiet()
    return f"quiet:{tag}" if tag else None


def hold_quiet(on: bool) -> None:
    """Hold (or drop) the daemon's QUIET.lock line, with its Windows PID; other lines are kept as they are."""
    keep = [x for x in _lines(fg.QUIET) if x.split()[0] != TAG]
    if on:
        keep.append(f"{TAG} {stamp()} pid={os.getpid()}")
    if keep:
        fg.QUIET.write_text("".join(x + "\n" for x in keep), encoding="utf-8", newline="\n")
    else:
        fg.QUIET.unlink(missing_ok=True)


def gpu_allowed(lines: list[str] | None = None) -> bool:
    """The card may be used: GPU_OK is absent or empty, or its first line is one of ours."""
    lines = _lines(GPU_OK) if lines is None else lines
    return not lines or lines[0] in MY_TAGS


def take_gpu(tag: str) -> bool:
    """Put `tag` in GPU_OK beside our other line, only while the first line is absent or ours."""
    lines = _lines(GPU_OK)
    if not gpu_allowed(lines):
        return False
    if tag not in lines:
        lead = 0
        while lead < len(lines) and lines[lead] in MY_TAGS:
            lead += 1
        lines.insert(lead, tag)
        GPU_OK.write_text("".join(x + "\n" for x in lines), encoding="utf-8", newline="\n")
    return True


def drop_gpu(tag: str) -> None:
    if not GPU_OK.exists():
        return
    lines = [x for x in _lines(GPU_OK) if x != tag]
    if lines:
        GPU_OK.write_text("".join(x + "\n" for x in lines), encoding="utf-8", newline="\n")
    else:
        GPU_OK.unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------------------- the state


class Loop:
    """The loop's folder: the start set, the buffer, the checkpoints, the players, the gate's games, the state and the log."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.starts = self.root / "starts"
        self.incoming = self.root / "buffer" / "incoming"
        self.shards = self.root / "buffer" / "shards"
        self.checkpoints = self.root / "checkpoints"
        self.players = self.root / "players"
        self.gates = self.root / "gate"
        self.games = self.root / "games"
        self.board = self.root / "board"
        self.trainer = self.root / "trainer"
        for d in (self.starts, self.incoming, self.shards, self.checkpoints, self.players, self.gates, self.games, self.board,
                  self.trainer):
            d.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.state = read_json(self.state_path, {}) or {}
        self.state.setdefault("segment", 0)
        self.state.setdefault("next_j", {})
        self.state.setdefault("kind_s", {})
        self.state.setdefault("gated", {})
        self.state.setdefault("versions", {"v0": {"dir": str(INCUMBENT0), "source": "agent gen7/candidate-r1 (M2's)", "n": 0}})
        self.state.setdefault("incumbent", "v0")
        self.state.setdefault("promotions", [])
        self._lock = threading.Lock()

    def save(self) -> None:
        with self._lock:
            write_json(self.state_path, self.state)

    def log(self, msg: str) -> None:
        line = f"{stamp()} {clean(msg)}\n"
        with self._lock, open(self.root / "loop.log", "a", encoding="utf-8") as fh:
            fh.write(line)

    def player_dir(self, version: str) -> Path:
        return Path(self.state["versions"][version]["dir"])

    def incumbent(self) -> tuple[str, Path]:
        v = self.state["incumbent"]
        return v, self.player_dir(v)

    def newest_checkpoint(self) -> tuple[str, Path] | None:
        done = sorted(p for p in self.checkpoints.glob("c[0-9][0-9][0-9][0-9]") if (p / "model.json").is_file()
                      and (p / "combat_policy.pt").is_file())
        return (done[-1].name, done[-1]) if done else None

    def generation_player(self) -> tuple[str, Path]:
        """The newest checkpoint, or the incumbent before the first: players reload between fights (between segments here)."""
        return self.newest_checkpoint() or self.incumbent()


# ---------------------------------------------------------------------------------------------------------- the start set


def board_run_seeds() -> set[str]:
    """Every run seed behind M2's starts (the board's 200 Act-1 bosses and their runs' Act-2 fights): never a training start."""
    return {s["seed"] for s in fg.read_jsonl(BOARD_DIR / "starts.jsonl")}


def fight_floors(run: dict) -> list[dict]:
    """The fights a night's run entered in Acts 1 and 2 (hallway, elite, boss; not events), up to the floor it reached."""
    nodes, floor = run.get("nodes") or "", int(run["floor"])
    out = []
    for i, ch in enumerate(nodes[: LAST_ACT2_INDEX + 1]):
        if ch not in NODE_KIND or i + 2 > floor:
            continue
        act = 1 if i <= fg.ACT1_BOSS_INDEX else 2
        out.append({"floor": i + 2, "act": act, "kind": NODE_KIND[ch], "cat": f"act{act}-{NODE_KIND[ch]}"})
    return out


def fights_per_run(runs: list[dict]) -> dict[str, float]:
    """Mean fights of each kind a run enters (a run that dies early enters fewer of the later ones)."""
    c: Counter = Counter()
    for r in runs:
        c.update(f["cat"] for f in fight_floors(r))
    n = max(1, len(runs))
    return {k: c[k] / n for k in KINDS}


def pick_runs(runs: list[dict], n: int, seed: int, exclude: set[str]) -> list[dict]:
    ok = [r for r in fg._eligible(runs) if r["seed"] not in exclude]  # noqa: SLF001
    return random.Random(seed).sample(ok, min(n, len(ok)))


def build_starts(loop: Loop, nights: tuple[str, ...] = NIGHTS, runs_per_night: int = 50, procs: int = 6, seed: int = 20261030,
                 agent_root: Path = REF / "agent") -> dict:
    """Replay `runs_per_night` runs of each night's emulator arm exactly (the M2 gym's `_replay_task`, the night's own agent) and keep
    every Act 1-2 fight they entered as a start. Resumable. Writes starts/starts.jsonl, its byte index and meta.json."""
    sys.path.insert(0, str(ROOT))
    from tools.solver import coverage  # noqa: PLC0415

    exclude = board_run_seeds()
    meta = {"nights": {}, "exclude_runs": len(exclude), "seed": seed, "runs_per_night": runs_per_night}
    all_runs: list[dict] = []
    for night in nights:
        src = LAB / "nightly" / night
        summ, runs, agent_dir, engine = coverage._arm(str(src))  # noqa: SLF001
        if engine != ENGINE:
            loop.log(f"starts: {night} played on {engine}, not {ENGINE}; skipped")
            continue
        all_runs += fg._eligible(runs)  # noqa: SLF001
        picks = pick_runs(runs, runs_per_night, seed + len(meta["nights"]), exclude)
        label = night.replace("-emu", "")
        target = loop.starts / f"replays-{label.replace('/', '-')}.jsonl"
        done = {r["seed"] for r in fg.read_jsonl(target)}
        tasks = []
        for r in picks:
            if r["seed"] in done:
                continue
            wanted = [{"id": f"{label}:{r['seed']}|f{f['floor']:02d}", "cat": f["cat"], "kind": f["kind"], "act": f["act"],
                       "floor": f["floor"], "seed": r["seed"], "ix": int(r["ix"]), "night": label} for f in fight_floors(r)]
            if wanted:
                tasks.append(("replay", r["seed"], int(r["ix"]), summ.get("config") or "a3-v1", wanted, r))
        meta["nights"][label] = {"agent": agent_dir, "runs": len(picks), "todo": len(tasks)}
        if tasks:
            loop.log(f"starts: replaying {len(tasks)} runs of {label}")
            stopped = guarded(loop, lambda t=tasks, a=agent_dir, tg=target: fg.run_tasks(
                t, fg._replay_task, tg, procs, init=fg._init_worker, init_args=(ENGINE, str(agent_root), a, str(loop.root)),  # noqa: SLF001
                mem_limit=fg.MEM_LIMIT_GB * 2**30, stop=lambda: why_blocked(loop.root, starting=False)))
            if stopped:
                return {"stopped": stopped}
    # assemble: one line per start (the document included), and a byte index so a process reads only the starts it plays
    starts = []
    for target in sorted(loop.starts.glob("replays-*.jsonl")):
        for r in fg.read_jsonl(target):
            for s in r.get("starts") or []:
                # the replay's own kind check knows elites and bosses only: a hallway fight is a monster room of its act
                s["kind_same"] = s.get("node_type") == NODE_TYPE[s["kind"]] and int(s.get("act", 0)) == int(s["cat"][3])
                if s.get("doc") and s["kind_same"]:
                    s["split"] = "val" if is_val_seed(s["seed"]) else "train"
                    starts.append(s)
    write_start_set(loop.starts, starts)
    meta.update({"starts": len(starts), "by_cat": dict(Counter(s["cat"] for s in starts)),
                 "by_split": dict(Counter(s["split"] for s in starts)), "fights_per_run": fights_per_run(all_runs),
                 "runs_counted": len(all_runs), "written": stamp()})
    write_json(loop.starts / "meta.json", meta)
    loop.log(f"starts: {len(starts)} starts {meta['by_cat']}")
    return meta


def write_start_set(d: Path, starts: list[dict]) -> None:
    """starts.jsonl (one start a line, its document included, sorted by id), its byte index and the list without documents."""
    starts = sorted(starts, key=lambda s: s["id"])
    index: dict[str, int] = {}
    out = Path(d) / "starts.jsonl"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        for s in starts:
            index[s["id"]] = fh.tell()
            fh.write((json.dumps(s) + "\n").encode("utf-8"))
    os.replace(tmp, out)
    write_json(Path(d) / "starts.index.json", index)
    write_json(Path(d) / "starts.lite.json", [{k: v for k, v in s.items() if k != "doc"} for s in starts])


_START_CACHE: dict = {}


def load_start(starts_path: str, sid: str) -> dict:
    """One start's line from the start set, through its byte index (a process never parses the whole set)."""
    cache = _START_CACHE.setdefault(starts_path, {"index": None, "docs": {}})
    if sid in cache["docs"]:
        return cache["docs"][sid]
    p = Path(starts_path)
    if cache["index"] is None:
        idx = p.with_name(p.stem + ".index.json")
        cache["index"] = read_json(idx) if idx.is_file() else None
    if cache["index"] is None:  # a small set without an index (the board's): read it whole once
        cache["docs"] = {s["id"]: s for s in fg.read_jsonl(p)}
        cache["index"] = {}
        return cache["docs"][sid]
    with open(p, "rb") as fh:
        fh.seek(cache["index"][sid])
        s = json.loads(fh.readline().decode("utf-8"))
    if len(cache["docs"]) > 256:
        cache["docs"].clear()
    cache["docs"][sid] = s
    return s


# ---------------------------------------------------------------------------------------------------------- the games (child side)

_LAST: list = [None]


def _install_capture() -> None:
    """Keep the result of the search that just returned, for the recorder: `fight_gym.play_steps` calls its grader right after the
    search, with no step in between, so the slot holds that game's result. The search itself is unchanged (the wrapper only reads)."""
    import live_search  # noqa: PLC0415

    if getattr(live_search.search_steps, "_gym_loop", False):
        return
    inner = live_search.search_steps

    def search_steps(handle, **kw):
        res = yield from inner(handle, **kw)
        _LAST[0] = res
        return res

    search_steps._gym_loop = True  # type: ignore[attr-defined]
    live_search.search_steps = search_steps


class Recorder:
    """`play_steps`'s grader: records each decision's training sample, and grades it with the exact solver where asked. It never
    changes the move: it reads the position and the search's result only."""

    def __init__(self, probe: bool, exact_wall_s: float = 0.0) -> None:
        import gym_lockstep  # noqa: PLC0415

        self.probe = probe
        self.grader = gym_lockstep.Grader(exact_wall_s) if exact_wall_s else None
        self.rows: list[dict] = []

    def __call__(self, world, k: int, best) -> dict:
        import gen_loop  # noqa: PLC0415
        import live_search  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        import prior_client  # noqa: PLC0415

        from sts2_gym import native  # noqa: PLC0415

        res, _LAST[0] = _LAST[0], None
        if res is None or res.best is not best:
            raise RuntimeError("gym_loop: the search result of this decision was not captured (play_steps changed?)")
        probe = world.fork(live_search._PROBE_SEED)  # noqa: SLF001 -- the root prior reads this copy
        try:
            tokens = prior_client.encode_position(probe.handle, self.probe)
        finally:
            probe.close()
        visits, q = gen_loop.root_targets(res.children, MAX_ACTIONS)
        mask = native.valid_actions(world.handle, MAX_ACTIONS)
        legal = np.array([1 if mask[a] else 0 for a in range(MAX_ACTIONS)], dtype=np.int8)
        g = self.grader(world, k, best) if self.grader is not None else {"k": k}
        exact = bool(g.get("exact")) and g.get("value") is not None and "error" not in g
        sb = (g.get("solver_best") or [-1, -1])[0]
        self.rows.append({"tokens": tokens, "visits": visits, "qvals": q, "legal": legal, "sims": int(res.sims), "k": k,
                          "exact": exact, "exact_value": float(g["value"]) if exact else float("nan"),
                          "q_played": float(g["q_played"]) if exact and g.get("q_played") is not None else float("nan"),
                          "solver_best": int(sb) if exact and sb is not None else -1, "played": int(best.action)})
        return g


def samples_arrays(rows: list[dict]) -> dict:
    import numpy as np  # noqa: PLC0415

    return {
        "types": np.concatenate([r["tokens"].types for r in rows]).astype(np.int16),
        "ids": np.concatenate([r["tokens"].ids for r in rows]).astype(np.int32),
        "owners": np.concatenate([r["tokens"].owners for r in rows]).astype(np.int16),
        "feats": np.concatenate([r["tokens"].feats for r in rows]).astype(np.float32),
        "lengths": np.array([len(r["tokens"].types) for r in rows], dtype=np.int32),
        "visits": np.stack([r["visits"] for r in rows]).astype(np.float32),
        "qvals": np.stack([r["qvals"] for r in rows]).astype(np.float32),
        "legal": np.stack([r["legal"] for r in rows]).astype(np.int8),
        "sims": np.array([r["sims"] for r in rows], dtype=np.int32),
        "k": np.array([r["k"] for r in rows], dtype=np.int16),
        "exact": np.array([r["exact"] for r in rows], dtype=np.int8),
        "exact_value": np.array([r["exact_value"] for r in rows], dtype=np.float32),
        "q_played": np.array([r["q_played"] for r in rows], dtype=np.float32),
        "solver_best": np.array([r["solver_best"] for r in rows], dtype=np.int16),
        "played": np.array([r["played"] for r in rows], dtype=np.int16),
    }


def save_npz(path: Path, arrays: dict) -> None:
    """Atomic compressed npz: a reader never sees half a file."""
    import numpy as np  # noqa: PLC0415

    path = Path(path)
    tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def player_config(name: str):
    """The search config a player plays under: a gym arm (a3-v1, x4, x16) or a search_configs name (the judge's, once it lands)."""
    if name in fg.ARMS:
        return fg.arm_settings(name)["cfg"]
    import search_configs  # noqa: PLC0415

    return search_configs.get(name)


def fight_key(sid: str, j: int, player: str) -> str:
    return f"{player}-" + hashlib.sha256(f"{sid}|{j}".encode()).hexdigest()[:16]


def gen_game_steps(task: tuple, clock, game, ctx):
    """One generation fight, as `gym_lockstep`'s game steps: `fight_gym.play_steps` under the teacher config, every decision
    recorded; the samples go to the incoming folder as one file, the row (without the per-decision grades) to the segment's file.
    task: (arm for the deadline, start id, j, cat, info) with info {starts, incoming, player, config, exact_wall_s, record, ns}."""
    import numpy as np  # noqa: PLC0415

    arm, sid, j, cat, info = task
    start = load_start(info["starts"], sid)
    rec = None
    if info.get("record", True):
        _install_capture()
        prior = ctx.extra.get("prior_fn")
        rec = Recorder(bool(getattr(prior, "probe", False)), float(info.get("exact_wall_s") or 0.0))
    world = fg._fresh_world(start["doc"], sid, j, info["ns"])  # noqa: SLF001
    try:
        row = yield from fg.play_steps(world, start, j, player_config(info["config"]), ctx.extra, info["ns"],
                                       deadline=fg.task_deadline(task), clock=clock, progress=game.note, grader=rec)
    finally:
        world.close()
    out = {"_task": [arm, sid, j], "arm": info.get("label", arm), "config": info["config"], "start": sid, "j": j, "cat": cat,
           "player": info["player"], "split": start.get("split"), "seed": start.get("seed"), "start_hp": start.get("hp"),
           "start_potions": start.get("potions"), "start_max_hp": start.get("max_hp"), **row}
    grades = out.pop("grades", None)
    if rec is not None:
        import gym_lockstep  # noqa: PLC0415

        if rec.grader is not None and grades:
            out["graded"] = gym_lockstep.grade_summary(grades)
        if rec.rows:
            arrays = samples_arrays(rec.rows)
            won = bool(row["won"])
            meta = {k: out.get(k) for k in ("start", "j", "cat", "player", "config", "split", "seed", "won", "end_hp", "max_hp",
                                            "L0", "capped", "potions_used", "turns", "decisions", "start_hp", "start_potions")}
            meta.update(encounter=start.get("encounter_id"), deck=start.get("deck_size"), relics=start.get("relics"),
                        label=int(row["end_hp"]) if won else 0, value_ok=not row.get("capped"))
            arrays["meta"] = np.array(json.dumps(meta))
            name = fight_key(sid, j, info["player"]) + ".npz"
            save_npz(Path(info["incoming"]) / name, arrays)
            out["samples"], out["sample_file"] = len(rec.rows), name
    return out


# ---------------------------------------------------------------------------------------------------------- the buffer


def compact(loop: Loop, segment: int) -> dict | None:
    """Merge the incoming fight files into one shard (`buffer/shards/sNNNNNN.npz`) and index it; the fight files go. Run after a
    segment and at start-up (a killed segment's finished fights are kept)."""
    import numpy as np  # noqa: PLC0415

    files = sorted(loop.incoming.glob("*.npz"))
    files = [f for f in files if ".tmp" not in f.name]
    for f in loop.incoming.glob("*.tmp*"):
        f.unlink(missing_ok=True)
    if not files:
        return None
    parts, metas, rows_of = [], [], []
    for f in files:
        try:
            with np.load(f) as z:
                a = {k: z[k] for k in z.files}
        except (OSError, ValueError, EOFError):
            loop.log(f"buffer: {f.name} unreadable; dropped")
            f.unlink(missing_ok=True)
            continue
        metas.append(json.loads(str(a.pop("meta"))))
        rows_of.append(int(a["lengths"].size))
        parts.append(a)
    if not parts:
        return None
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    n = sum(rows_of)
    merged["fight"] = np.repeat(np.arange(len(parts), dtype=np.int32), rows_of)
    merged["label"] = np.repeat(np.array([m["label"] for m in metas], dtype=np.int16), rows_of)
    merged["value_ok"] = np.repeat(np.array([bool(m["value_ok"]) for m in metas], dtype=np.int8), rows_of)
    merged["val"] = np.repeat(np.array([m.get("split") == "val" for m in metas], dtype=np.int8), rows_of)
    merged["fights_meta"] = np.array(json.dumps(metas))
    name = f"s{segment:06d}-{int(time.time())}.npz"
    save_npz(loop.shards / name, merged)
    entry = {"shard": name, "segment": segment, "rows": n, "fights": len(parts), "bytes": (loop.shards / name).stat().st_size,
             "players": dict(Counter(m["player"] for m in metas)), "cats": dict(Counter(m["cat"] for m in metas)),
             "val_rows": int(merged["val"].sum()), "t": stamp()}
    with open(loop.root / "buffer" / "index.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    for f in files:
        f.unlink(missing_ok=True)
    return entry


def retain(loop: Loop, max_rows: int, max_days: float, max_gb: float) -> dict:
    """The buffer's retention window and disk cap: the newest shards are kept while their rows stay within `max_rows`, their age
    within `max_days` and their bytes within `max_gb`; older shards are deleted (the oldest first)."""
    shards = sorted(loop.shards.glob("s*.npz"), key=lambda p: p.name, reverse=True)
    index = {e["shard"]: e for e in fg.read_jsonl(loop.root / "buffer" / "index.jsonl")}
    rows = size = 0
    gone = []
    now = time.time()
    for p in shards:
        if ".tmp" in p.name:
            continue
        e = index.get(p.name, {})
        r, b = int(e.get("rows", 0)), p.stat().st_size
        age_d = (now - p.stat().st_mtime) / 86400
        if rows + r > max_rows or size + b > max_gb * 2**30 or age_d > max_days:
            p.unlink(missing_ok=True)
            gone.append(p.name)
            continue
        rows, size = rows + r, size + b
    if gone:
        loop.log(f"buffer: retention removed {len(gone)} shards; kept {rows} rows, {size / 2**30:.2f} GB")
    return {"rows": rows, "gb": round(size / 2**30, 3), "removed": len(gone)}


def prune_checkpoints(loop: Loop, keep: int = 48) -> None:
    """The newest `keep` checkpoints stay (a promoted one is copied to players/ first, so it never goes with them)."""
    done = sorted(loop.checkpoints.glob("c[0-9][0-9][0-9][0-9]"))
    for p in done[:-keep] if keep > 0 else []:
        shutil.rmtree(p, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------- the weights


def kind_weights(loop: Loop, procs: int = 6, per_kind: int = 50, seeds: int = 2, k: int = 16, prior: str = "cpu") -> dict:
    """The start mix (gym-build P2): per kind, the fights a run enters times the mean L0 a fight loses (a death loses all of it),
    the loss measured on `per_kind` training starts x `seeds` games at a3-v1 with the incumbent. Resumable; writes weights.json."""
    meta = read_json(loop.starts / "meta.json", {})
    lite = read_json(loop.starts / "starts.lite.json", [])
    by: dict[str, list[dict]] = defaultdict(list)
    for s in lite:
        if s["split"] == "train":
            by[s["cat"]].append(s)
    rnd = random.Random(7)
    pick = {c: rnd.sample(v, min(per_kind, len(v))) for c, v in sorted(by.items())}
    target = loop.starts / "weights_games.jsonl"
    done = {(r["start"], int(r["j"])) for r in fg.read_jsonl(target)}
    inc, inc_dir = loop.incumbent()
    info = {"starts": str(loop.starts / "starts.jsonl"), "incoming": str(loop.incoming), "player": inc, "config": "a3-v1",
            "record": False, "ns": NS_WEIGHTS}
    tasks = [("a3-v1", s["id"], j, s["cat"], info) for c in pick.values() for s in c for j in range(seeds) if (s["id"], j) not in done]
    if tasks:
        stopped = play_tasks(loop, tasks, target, inc_dir, procs, k, prior, "weights")
        if stopped:
            return {"stopped": stopped}
    rows = fg.read_jsonl(target)
    loss: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        start_l0 = fg.l0(True, int(r.get("start_hp") or 0), int(r.get("start_potions") or 0), int(r.get("start_max_hp") or 1))
        loss[r["cat"]].append(start_l0 - float(r.get("L0", 0.0)))
    fpr = meta.get("fights_per_run") or {}
    raw = {c: fpr.get(c, 0.0) * (sum(v) / len(v)) for c, v in loss.items() if v}
    tot = sum(raw.values()) or 1.0
    w = {"weights": {c: raw[c] / tot for c in sorted(raw)}, "fights_per_run": fpr,
         "mean_l0_lost": {c: sum(v) / len(v) for c, v in sorted(loss.items())}, "games": {c: len(v) for c, v in sorted(loss.items())},
         "win": {c: sum(1 for r in rows if r["cat"] == c and r.get("won")) / max(1, len(loss[c])) for c in sorted(loss)},
         "player": inc, "config": "a3-v1", "written": stamp()}
    write_json(loop.starts / "weights.json", w)
    loop.log("weights: " + ", ".join(f"{c} {x:.3f}" for c, x in w["weights"].items()))
    return w


# ---------------------------------------------------------------------------------------------------------- playing


def guarded(loop: Loop, fn):
    """Run one heavy step with the daemon's QUIET.lock line held (refreshed by the step's stop polls), then drop it."""
    hold_quiet(True)
    try:
        return fn()
    finally:
        hold_quiet(False)


def _champion(player_dir: Path):
    import combat_prior  # noqa: PLC0415

    return combat_prior.find(player_dir)


class Server:
    """The prior server for one player (P1's, on the card), started only while GPU_OK allows it; None means the CPU prior."""

    def __init__(self, loop: Loop, player_dir: Path, name: str) -> None:
        self.loop, self.handle, self.opts = loop, None, None
        if os.environ.get("GYM_LOOP_NO_GPU") or not take_gpu(TAG):
            return
        from prior_launch import start_prior_server  # noqa: PLC0415

        champ = _champion(player_dir)
        self.handle = start_prior_server(GPU_PYTHON, champ.policy, lib_path=ROOT / "out", kill_file=GPU_OFF,
                                         stats_path=loop.games / f"server-{name}.json", priority="below")
        if self.handle is None:
            drop_gpu(TAG)
            loop.log("server: did not start; the CPU prior plays")
            return
        self.opts = {"endpoint": self.handle.endpoint, "authkey": self.handle.authkey, "policy_sha256": champ.policy_sha256}

    @property
    def pid(self) -> int | None:
        return self.handle.proc.pid if self.handle is not None and self.handle.proc is not None else None

    def stop(self) -> dict:
        stats = {}
        if self.handle is not None:
            stats = self.handle.read_stats()
            self.handle.stop()
            self.handle = None
            drop_gpu(TAG)
        return stats


def play_tasks(loop: Loop, tasks: list, target: Path, player_dir: Path, procs: int, k: int, prior: str, what: str,
               steps: str = "gym_loop:gen_game_steps", daemon=None, extra_opts: dict | None = None) -> str | None:
    """Play `tasks` in P1's lockstep processes (prior on the card when allowed), rows appended to `target`, under the QUIET line.
    Returns why it stopped, or None when every task has its row."""
    import gym_lockstep  # noqa: PLC0415

    init_args = (ENGINE, str(player_dir.parent), player_dir.name, str(loop.root))
    server = Server(loop, player_dir, what) if prior == "server" else None
    opts = {"prior": "server" if server is not None and server.opts else "cpu", "server": server.opts if server else None,
            "game_steps": steps, "worlds": "reload", "grade_wall_s": 0, "paths": [str(ROOT / "rl")], **(extra_opts or {})}
    if daemon is not None:
        daemon.server_pid = server.pid if server else None
    t0 = time.perf_counter()
    last = [time.time()]

    def stop() -> str | None:
        if time.time() - last[0] > 1800:
            hold_quiet(True)  # refresh the line every half hour while the step runs
            last[0] = time.time()
        return why_blocked(loop.root, starting=False)

    try:
        hold_quiet(True)
        stopped, stats = gym_lockstep.run_lockstep(tasks, target, procs, min(k, gym_lockstep.K_MAX), init_args, opts, stop=stop,
                                                   log=lambda m: loop.log(f"{what}: {m}"))
    finally:
        hold_quiet(False)
        sstats = server.stop() if server is not None else {}
        if daemon is not None:
            daemon.server_pid = None
    wall = time.perf_counter() - t0
    with open(loop.games / "steps.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"t": stamp(), "what": what, "tasks": len(tasks), "wall_s": round(wall, 1), "stopped": stopped,
                             "prior": opts["prior"], "procs": procs, "k": k, "server": {x: sstats.get(x) for x in
                             ("replays", "rows", "replay_ms_mean", "busy_s") if x in sstats},
                             "processes": [{x: s.get(x) for x in ("games", "rounds", "requests", "wall_s", "net_s", "games_s",
                                                                  "cpu_s")} for s in stats]}) + "\n")
    return stopped


# ---------------------------------------------------------------------------------------------------------- generation


def sample_tasks(loop: Loop, player: str, budget_s: float, teacher: str, exact_wall_s: float, rnd: random.Random) -> list:
    """A segment's fights: kinds by the start mix's weights, a start uniformly within the kind, game j the start's next unplayed
    one; until the estimated own seconds reach `budget_s`. Long fights first, so the segment's tail is the short ones."""
    weights = (read_json(loop.starts / "weights.json", {}) or {}).get("weights") or {}
    lite = read_json(loop.starts / "starts.lite.json", [])
    by: dict[str, list[str]] = defaultdict(list)
    for s in lite:
        by[s["cat"]].append(s["id"])
    cats = [c for c in KINDS if by.get(c) and weights.get(c, 0) > 0]
    if not cats:
        raise RuntimeError("no start mix: run `gym_loop.py starts` and `weights` first")
    w = [weights[c] for c in cats]
    info = {"starts": str(loop.starts / "starts.jsonl"), "incoming": str(loop.incoming), "player": player, "config": teacher,
            "exact_wall_s": exact_wall_s, "record": True, "ns": NS_GEN}
    nj = loop.state["next_j"]
    tasks, spent = [], 0.0
    while spent < budget_s and len(tasks) < 5000:
        c = rnd.choices(cats, w)[0]
        sid = rnd.choice(by[c])
        j = int(nj.get(sid, 0))
        nj[sid] = j + 1
        tasks.append((teacher, sid, j, c, info))
        spent += est_seconds(loop, c)
    tasks.sort(key=lambda t: -est_seconds(loop, t[3]))
    return tasks


def est_seconds(loop: Loop, cat: str) -> float:
    s, n = loop.state["kind_s"].get(cat, [0.0, 0])
    return s / n if n >= 5 else MEAN_X16_S.get(cat, 30.0)


def learn_seconds(loop: Loop, rows: list[dict]) -> None:
    for r in rows:
        if r.get("cat") and r.get("seconds") is not None and not r.get("capped"):
            s, n = loop.state["kind_s"].get(r["cat"], [0.0, 0])
            loop.state["kind_s"][r["cat"]] = [s + float(r["seconds"]), n + 1]


# ---------------------------------------------------------------------------------------------------------- the gate


def mcnemar_one_sided(b: int, c: int) -> float:
    """P(at least b candidate-only wins of the b + c discordant pairs | no difference): the exact one-sided McNemar test."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, sum(math.comb(n, i) for i in range(b, n + 1)) / 2**n)


def paired(cand: list[dict], inc: list[dict], seeds: int) -> dict:
    """The pairs (start, j < seeds) both arms played: wins (a capped game is lost), discordant counts, L0 difference."""
    a = {(r["start"], int(r["j"])): r for r in cand if int(r["j"]) < seeds}
    b = {(r["start"], int(r["j"])): r for r in inc if int(r["j"]) < seeds}
    keys = sorted(set(a) & set(b))
    wa = [bool(a[x].get("won")) and not a[x].get("capped") for x in keys]
    wb = [bool(b[x].get("won")) and not b[x].get("capped") for x in keys]
    la = [float(a[x].get("L0", 0.0)) for x in keys]
    lb = [float(b[x].get("L0", 0.0)) for x in keys]
    n = len(keys)
    d_win = [float(x) - float(y) for x, y in zip(wa, wb)]
    d_l0 = [x - y for x, y in zip(la, lb)]
    mw, sw, _ = fg.mean_se(d_win)
    ml, sl, _ = fg.mean_se(d_l0)
    return {"pairs": n, "cand_wins": sum(wa), "inc_wins": sum(wb), "cand_only": sum(1 for x, y in zip(wa, wb) if x and not y),
            "inc_only": sum(1 for x, y in zip(wa, wb) if y and not x), "win_diff": mw, "win_se": sw, "l0_diff": ml, "l0_se": sl,
            "cand_rate": sum(wa) / n if n else None, "inc_rate": sum(wb) / n if n else None,
            "capped": sum(1 for x in keys if a[x].get("capped")) + sum(1 for x in keys if b[x].get("capped"))}


def decide(cand: list[dict], inc: list[dict], stage: int, rule: dict | None = None) -> dict:
    """The fixed rule on the games so far. Stage 1: 'continue' or 'reject' (futility). Stage 2: 'promote' or 'reject'."""
    rule = rule or PROMOTION_RULE
    if stage == 1:
        p = paired(cand, inc, rule["stage1_seeds"])
        need = rule["starts"] * rule["stage1_seeds"]
        if p["pairs"] < need:
            return {"verdict": "incomplete", "stage": 1, **p}
        go = p["win_diff"] is not None and 100 * p["win_diff"] >= rule["futility_points"]
        return {"verdict": "continue" if go else "reject", "stage": 1, "why": None if go else
                f"stage 1: {100 * (p['win_diff'] or 0):+.1f} points, under the {rule['futility_points']:+.1f} needed to go on", **p}
    p = paired(cand, inc, rule["seeds"])
    if p["pairs"] < rule["starts"] * rule["seeds"]:
        return {"verdict": "incomplete", "stage": 2, **p}
    pval = mcnemar_one_sided(p["cand_only"], p["inc_only"])
    ok_l0 = (p["l0_diff"] or 0.0) >= 0.0 or not rule["l0_not_worse"]
    ok = pval < rule["alpha"] and ok_l0
    why = None if ok else ("McNemar p %.3f >= %.2f" % (pval, rule["alpha"]) if pval >= rule["alpha"] else "mean L0 lower")
    return {"verdict": "promote" if ok else "reject", "stage": 2, "p": pval, "why": why, **p}


def board_starts() -> list[dict]:
    return [s for s in fg.read_jsonl(BOARD_DIR / "starts.jsonl") if s["cat"] == PROMOTION_RULE["cat"]][: PROMOTION_RULE["starts"]]


def gate_tasks(version: str, config: str, seeds: range, done: set, label: str) -> list:
    info = {"starts": str(BOARD_DIR / "starts.jsonl"), "incoming": "", "player": version, "config": config, "record": False,
            "ns": PROMOTION_RULE["ns"], "label": label}
    return [("a3-v1", s["id"], j, s["cat"], info) for j in seeds for s in board_starts() if (s["id"], j) not in done]


def engine_tag() -> str:
    return sha256_file(ROOT / "out" / "Sts2Emulator.dll")[:12]


def run_gate(loop: Loop, cand: str, cand_dir: Path, config: str, procs: int, k: int, prior: str, daemon=None,
             judge: str | None = None) -> dict:
    """Score a checkpoint against the incumbent with the fixed rule (resumable: the games on disk are kept)."""
    inc, inc_dir = loop.incumbent()
    eng = engine_tag()
    # the player is a parameter: its search config, and the judge checkpoint a judge config's leaves read (passed to the
    # lockstep processes as gym_lockstep's `judge` option, on the CPU); the incumbent's kept games are per config and judge
    ctag = config + (f"+j{sha256_file(Path(judge))[:8]}" if judge else "")
    xo = {"judge": {"path": str(judge)}} if judge else None
    inc_file = loop.gates / f"inc-{inc}-{ctag}-{eng}.jsonl"
    cand_file = loop.gates / f"{cand}-{ctag}-{eng}.jsonl"
    rule = PROMOTION_RULE
    out: dict = {"candidate": cand, "incumbent": inc, "config": config, "engine": eng, "rule": rule["name"]}
    for stage, seeds in ((1, range(rule["stage1_seeds"])), (2, range(rule["seeds"]))):
        for who, d, f in ((inc, inc_dir, inc_file), (cand, cand_dir, cand_file)):
            done = {(r["start"], int(r["j"])) for r in fg.read_jsonl(f)}
            tasks = gate_tasks(who, config, seeds, done, label=who)
            if tasks:
                loop.log(f"gate {cand}: stage {stage}, {who} plays {len(tasks)} games")
                stopped = play_tasks(loop, tasks, f, d, procs, k, prior, f"gate-{who}", daemon=daemon, extra_opts=xo)
                if stopped:
                    return {**out, "verdict": "stopped", "why": stopped}
        v = decide(fg.read_jsonl(cand_file), fg.read_jsonl(inc_file), stage)
        out.update(v)
        loop.log(f"gate {cand}: stage {stage} {v['verdict']}: cand {v.get('cand_wins')} inc {v.get('inc_wins')} of {v.get('pairs')}"
                 f" ({100 * (v.get('win_diff') or 0):+.1f} points, SE {100 * (v.get('win_se') or 0):.1f})"
                 + (f" p {v['p']:.3f}" if "p" in v else ""))
        if v["verdict"] != "continue":
            break
    out["cand_file"], out["inc_file"] = cand_file.name, inc_file.name
    return out


def promote(loop: Loop, cand: str, cand_dir: Path, verdict: dict, board: str) -> str:
    """The checkpoint becomes the next version: copied to players/vN, the incumbent from now on, its board row built."""
    n = 1 + max(int(v[1:]) for v in loop.state["versions"])
    ver = f"v{n}"
    dest = loop.players / ver
    if not dest.exists():
        shutil.copytree(cand_dir, dest)
    loop.state["versions"][ver] = {"dir": str(dest), "source": f"checkpoint {cand}", "n": n, "promoted": stamp(),
                                   "verdict": {k: verdict.get(k) for k in ("cand_wins", "inc_wins", "pairs", "win_diff", "win_se",
                                                                           "l0_diff", "p", "cand_rate", "inc_rate")}}
    prev = loop.state["incumbent"]
    loop.state["incumbent"] = ver
    loop.state["promotions"].append({"version": ver, "checkpoint": cand, "over": prev, "t": stamp()})
    loop.save()
    rows = fg.read_jsonl(loop.gates / verdict["cand_file"])
    with open(loop.board / f"fp-{ver}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({**{k: v for k, v in r.items() if not k.startswith("_")}, "arm": f"fp-{ver}"}) + "\n")
    loop.log(f"promoted {cand} as {ver} over {prev}: {verdict.get('cand_wins')} against {verdict.get('inc_wins')} wins of "
             f"{verdict.get('pairs')}")
    if board != "off":
        try:
            board_commit(loop, ver, verdict, commit=board == "commit")
        except Exception as error:  # noqa: BLE001 -- the board is retried at the next start (`board_pending`)
            loop.log(f"board: {type(error).__name__}: {error}")
    return ver


# ---------------------------------------------------------------------------------------------------------- the fight board

BOARD_REPO = Path(os.environ.get("GYM_LOOP_BOARD_REPO", "D:/spireline/spireline"))
BOARD_WT = Path(os.environ.get("GYM_LOOP_BOARD_WT", "D:/spireline/wt-board"))
BOARD_PAGE = "docs/paper/fight-board.html"
STANDING_ARMS = (("a3-v1", "Usual thinking"), ("x16", "Sixteen times the thinking"), ("x4", "Four times the thinking"))
COAUTHOR = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


def _git(*args: str, cwd: Path) -> str:
    r = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace",
                       check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        raise RuntimeError(clean(f"git {' '.join(args[:2])}: {r.stderr.strip()[-300:]}"))
    return r.stdout


def page_log(html: str) -> list[str]:
    m = re.search(r"const LOG = (\[.*?\]);", html, re.S)
    return json.loads(m.group(1)) if m else []


def board_line(ver: str, v: dict, eng_commit: str) -> str:
    """The board's changelog line, in plain language."""
    pts = 100 * (v.get("win_diff") or 0.0)
    return (f"{utc().strftime('%Y-%m-%d')}: fight player, version {ver[1:]} added, trained by the always-on loop on its own "
            f"practice fights; it won {v.get('cand_wins')} of {v.get('pairs')} replays against {v.get('inc_wins')} for the version "
            f"before it on the same fights and draws ({pts:+.1f} points); played on the copy of the game as of {eng_commit}")


def board_commit(loop: Loop, ver: str, verdict: dict, commit: bool = True) -> str:
    """The standing fight-board step: the page rebuilt with every promoted version as a row, committed to Spireline main from its
    own worktree (never the shared checkout), pushed; on a moved main, rebased and pushed again."""
    if not BOARD_WT.exists():
        _git("worktree", "add", "--detach", str(BOARD_WT), "origin/main", cwd=BOARD_REPO)
    _git("fetch", "-q", "origin", cwd=BOARD_WT)
    _git("checkout", "-q", "--detach", "origin/main", cwd=BOARD_WT)
    page = BOARD_WT / BOARD_PAGE
    old = page_log(page.read_text(encoding="utf-8"))
    vers = sorted((v for v in loop.state["versions"] if v != "v0"), key=lambda x: int(x[1:]))
    eng_commit = _git("rev-parse", "--short", "HEAD", cwd=ROOT).strip()
    line = board_line(ver, verdict, eng_commit)
    cmd = [sys.executable, "tools/fight_board.py", "--out", BOARD_PAGE, "--starts", str(BOARD_DIR / "starts.jsonl"),
           "--games", str(BOARD_DIR / "games.jsonl")]
    for v in vers:
        cmd += ["--games", str(loop.board / f"fp-{v}.jsonl")]
    for key, label in STANDING_ARMS:
        cmd += ["--arm", f"{key}={label}"]
    for v in vers:
        cmd += ["--arm", f"fp-{v}=Fight player, version {v[1:]}"]
    for x in [*old, line] if line not in old else old:
        cmd += ["--log", x]
    r = subprocess.run(cmd, cwd=str(BOARD_WT), capture_output=True, text=True, check=False,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        raise RuntimeError(clean("fight_board.py: " + r.stderr[-300:]))
    if not commit:
        shutil.copy(page, loop.board / "fight-board.html")
        return "dry"
    _git("add", BOARD_PAGE, cwd=BOARD_WT)
    v = verdict
    msg = (f"docs: fight board, fight player version {ver[1:]} added (always-on loop: {v.get('cand_wins')} of {v.get('pairs')} "
           f"Act-1 boss replays won against {v.get('inc_wins')}, {100 * (v.get('win_diff') or 0):+.1f} points, McNemar p "
           f"{v.get('p', float('nan')):.3f})\n\n{COAUTHOR}\n")
    _git("commit", "-q", "-m", msg, cwd=BOARD_WT)
    for _ in range(3):
        try:
            _git("push", "-q", "origin", "HEAD:main", cwd=BOARD_WT)
            break
        except RuntimeError:
            _git("fetch", "-q", "origin", cwd=BOARD_WT)
            _git("rebase", "-q", "origin/main", cwd=BOARD_WT)
    else:
        raise RuntimeError("board: push failed three times")
    h = _git("rev-parse", "--short", "HEAD", cwd=BOARD_WT).strip()
    loop.state.setdefault("board", {})[ver] = h
    loop.save()
    loop.log(f"board: {ver} committed {h}")
    return h


# ---------------------------------------------------------------------------------------------------------- the card's split


class GpuWatch:
    """Once a minute, the card's per-process busy share (Windows' GPU Engine counters, the busiest engine per process), split into
    the loop's inference (its prior server), its training (the trainer) and everyone else (the lab's jobs and the desktop). Only
    process ids and numbers are read; nothing of the counter's path (it names the machine) is kept."""

    _PID = re.compile(r"pid_(\d+)_")

    def __init__(self, loop: Loop, daemon) -> None:
        self.loop, self.daemon = loop, daemon
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if shutil.which("typeperf"):
            self.thread.start()

    def stop(self) -> None:
        self._stop.set()

    def sample(self) -> dict | None:
        r = subprocess.run(["typeperf", r"\GPU Engine(*)\Utilization Percentage", "-sc", "2"], capture_output=True, text=True,
                           check=False, timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        lines = [x for x in r.stdout.splitlines() if x.startswith('"')]
        if len(lines) < 3:
            return None
        head = next(iter(__import__("csv").reader([lines[0]])))
        vals = next(iter(__import__("csv").reader([lines[-1]])))
        per: dict[int, float] = defaultdict(float)
        for h, v in zip(head[1:], vals[1:]):
            m = self._PID.search(h)
            if not m:
                continue
            try:
                x = float(v)
            except ValueError:
                continue
            per[int(m.group(1))] = max(per[int(m.group(1))], x)
        names = _python_pids()
        inf = per.get(self.daemon.server_pid or -1, 0.0)
        tr = per.get(self.daemon.trainer_pid or -1, 0.0)
        mine = {self.daemon.server_pid, self.daemon.trainer_pid}
        lab = sum(x for p, x in per.items() if p in names and p not in mine)
        other = sum(x for p, x in per.items() if p not in names and p not in mine)
        return {"t": stamp(), "inference": round(inf, 1), "training": round(tr, 1), "lab_python": round(lab, 1),
                "other": round(other, 1), "phase": self.daemon.phase}

    def _run(self) -> None:
        while not self._stop.wait(60):
            with contextlib.suppress(Exception):
                s = self.sample()
                if s is not None:
                    with open(self.loop.root / "gpu.jsonl", "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(s) + "\n")


def _python_pids() -> set[int]:
    r = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, check=False,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    out = set()
    for row in __import__("csv").reader(r.stdout.splitlines()):
        if len(row) > 1 and row[0].lower().startswith("python"):
            with contextlib.suppress(ValueError):
                out.add(int(row[1]))
    return out


def gpu_split(loop: Loop, since: str | None = None) -> dict:
    rows = [r for r in fg.read_jsonl(loop.root / "gpu.jsonl") if since is None or r["t"] >= since]
    n = len(rows)
    if not n:
        return {"samples": 0}
    return {"samples": n, **{k: round(sum(r[k] for r in rows) / n, 1) for k in ("inference", "training", "lab_python", "other")}}


# ---------------------------------------------------------------------------------------------------------- the daemon


class Daemon:
    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.loop = Loop(a.root)
        self.phase = "start"
        self.server_pid: int | None = None
        self.trainer_pid: int | None = None
        self.trainer = None
        self.job = None
        self.commit = ""
        self.rnd = random.Random(int(time.time()))

    # -- status
    def status(self, phase: str, **kw) -> None:
        self.phase = phase
        doc = {"pid": os.getpid(), "phase": phase, "t": stamp(), "root": str(self.loop.root), "commit": self.commit,
               "segment": self.loop.state["segment"], "incumbent": self.loop.state["incumbent"],
               "server_pid": self.server_pid, "trainer_pid": self.trainer_pid, **kw}
        STATUS_DIR.mkdir(parents=True, exist_ok=True)
        write_json(STATUS_DIR / "status", doc)

    # -- the trainer process
    def start_trainer(self) -> None:
        if self.a.no_trainer or (self.trainer is not None and self.trainer.poll() is None):
            return
        if self.trainer is not None:  # it ended on its own: logged, and started again at most every ten minutes
            self.loop.log(f"trainer: ended rc={self.trainer.returncode}")
            self.trainer = None
            self.trainer_pid = None
        if time.time() - getattr(self, "_trainer_t", 0.0) < 600:
            return
        self._trainer_t = time.time()
        from tools.lab import winsys  # noqa: PLC0415

        inc, inc_dir = self.loop.incumbent()
        argv = [GPU_PYTHON, str(ROOT / "rl" / "gym_loop_train.py"), "--root", str(self.loop.root), "--base", str(inc_dir),
                "--parent-pid", str(os.getpid()), "--every-min", str(self.a.checkpoint_min), "--max-epochs", str(self.a.max_epochs),
                "--window", str(self.a.train_window)]
        logf = open(self.loop.trainer / "trainer.out", "a", encoding="utf-8")  # noqa: SIM115 -- the child's for its life
        self.trainer = winsys.suspended_popen(argv, self.job, stdout=logf, stderr=subprocess.STDOUT, cwd=str(ROOT),
                                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.trainer_pid = self.trainer.pid
        self.loop.log(f"trainer: started (pid {self.trainer_pid})")

    def stop_trainer(self) -> None:
        if self.trainer is None:
            return
        (self.loop.trainer / "STOP").write_text("stop\n", encoding="utf-8")
        try:
            self.trainer.wait(120)
        except subprocess.TimeoutExpired:
            self.trainer.kill()  # our own child, by its handle
        (self.loop.trainer / "STOP").unlink(missing_ok=True)
        self.trainer = None
        self.trainer_pid = None

    # -- the steps
    def next_gate(self) -> tuple[str, Path] | None:
        newest = self.loop.newest_checkpoint()
        if newest is None or newest[0] in self.loop.state["gated"]:
            return None
        return newest

    def gate(self, cand: str, cand_dir: Path) -> None:
        self.status("gate", candidate=cand)
        v = run_gate(self.loop, cand, cand_dir, self.a.player_config, self.a.procs, self.a.gate_k, self.a.prior, daemon=self,
                     judge=self.a.judge or None)
        if v.get("verdict") == "stopped":
            return
        older = sorted(p.name for p in self.loop.checkpoints.glob("c[0-9][0-9][0-9][0-9]") if p.name < cand)
        for old in older:  # superseded while untested: the newest is always the one scored
            self.loop.state["gated"].setdefault(old, {"verdict": "superseded", "by": cand})
        self.loop.state["gated"][cand] = {k: v.get(k) for k in ("verdict", "stage", "why", "pairs", "cand_wins", "inc_wins",
                                                                "win_diff", "win_se", "l0_diff", "p", "incumbent", "config")}
        self.loop.save()
        with open(self.loop.gates / "verdicts.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": stamp(), **{k: v.get(k) for k in v if k not in ("cand_file", "inc_file")}}) + "\n")
        if v.get("verdict") == "promote":
            promote(self.loop, cand, cand_dir, v, self.a.board)

    def segment(self) -> None:
        loop = self.loop
        loop.state["segment"] += 1
        seg = loop.state["segment"]
        player, pdir = loop.generation_player()
        tasks = sample_tasks(loop, player, self.a.procs * self.a.segment_min * 60, self.a.teacher, self.a.exact_wall_s, self.rnd)
        loop.save()
        target = loop.games / f"seg{seg:06d}.jsonl"
        self.status("generate", player=player, tasks=len(tasks))
        loop.log(f"segment {seg}: {len(tasks)} fights by {player} under {self.a.teacher}")
        t0 = time.time()
        stopped = play_tasks(loop, tasks, target, pdir, self.a.procs, self.a.k, self.a.prior, f"seg{seg}", daemon=self)
        rows = fg.read_jsonl(target)
        learn_seconds(loop, rows)
        entry = compact(loop, seg)
        loop.save()
        with open(loop.games / "segments.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": stamp(), "segment": seg, "player": player, "fights": len(rows), "tasks": len(tasks),
                                 "wall_s": round(time.time() - t0, 1), "stopped": stopped,
                                 "capped": sum(1 for r in rows if r.get("capped")), "samples": entry["rows"] if entry else 0,
                                 "cats": dict(Counter(r["cat"] for r in rows)),
                                 "won": dict(Counter(r["cat"] for r in rows if r.get("won")))}) + "\n")
        retain(loop, self.a.buffer_rows, self.a.buffer_days, self.a.buffer_gb)
        prune_checkpoints(loop)

    def wait_free(self) -> None:
        """Drop the line and let a waiting step in: wait `yield_s`, then while anything blocks a start."""
        hold_quiet(False)
        self.status("yield")
        end = time.time() + self.a.yield_s
        while time.time() < end and not (self.loop.root / "STOP").exists():
            time.sleep(5)
        why = why_blocked(self.loop.root, starting=True)
        while why and why != "stop":
            self.status("waiting", why=why)
            time.sleep(30)
            why = why_blocked(self.loop.root, starting=True)

    def run(self) -> int:
        from tools.lab import winsys  # noqa: PLC0415

        winsys.set_below_normal()
        if not winsys.acquire_mutex("spireline-gym-loop"):
            print("another gym loop is running")
            return 2
        self.job = winsys.JobObject(kill_on_close=True)  # everything started from here dies with the daemon, however it ends
        self.job.assign_self()
        self.commit = _git("rev-parse", "--short", "HEAD", cwd=ROOT).strip()
        (self.loop.root / "STOP").unlink(missing_ok=True)
        self.loop.log(f"daemon: start pid {os.getpid()} at {self.commit} player config {self.a.player_config}, teacher "
                      f"{self.a.teacher}, {self.a.procs} processes K={self.a.k}")
        watch = GpuWatch(self.loop, self)
        watch.start()
        self.status("start")
        entry = compact(self.loop, self.loop.state["segment"])  # a killed segment's finished fights
        if entry:
            self.loop.log(f"buffer: {entry['fights']} fights of an interrupted segment kept")
        rc = 0
        try:
            while not (self.loop.root / "STOP").exists():
                if self.a.until and stamp() >= self.a.until:
                    self.loop.log("daemon: --until reached")
                    break
                why = why_blocked(self.loop.root, starting=True)
                if why:
                    self.status("waiting", why=why)
                    self.start_trainer()  # the night is training time: the trainer keeps its own rules (low CPU, GPU_OK)
                    time.sleep(30)
                    continue
                if not (self.loop.starts / "starts.jsonl").is_file() or not (self.loop.starts / "meta.json").is_file():
                    self.status("starts")
                    r = build_starts(self.loop, runs_per_night=self.a.runs_per_night, procs=self.a.procs)
                    self.wait_free()
                    if r.get("stopped"):
                        continue
                    continue
                if not (self.loop.starts / "weights.json").is_file():
                    self.status("weights")
                    kind_weights(self.loop, procs=self.a.procs, prior=self.a.prior)
                    self.wait_free()
                    continue
                self.start_trainer()
                cand = self.next_gate()
                if cand is not None and not self.a.no_gate:
                    self.gate(*cand)
                else:
                    self.segment()
                self.wait_free()
        except Exception as error:  # noqa: BLE001 -- logged; the state on disk stays for the restart
            import traceback  # noqa: PLC0415

            self.loop.log("daemon: " + traceback.format_exc()[-1500:])
            rc = 1
        finally:
            hold_quiet(False)
            drop_gpu(TAG)
            self.stop_trainer()
            watch.stop()
            self.loop.save()
            self.status("stopped", rc=rc)
            self.loop.log(f"daemon: stopped rc={rc}")
        return rc


# ---------------------------------------------------------------------------------------------------------- commands


def cmd_stop(a) -> int:
    """Ask the daemon to stop (its STOP file); after `--wait-s`, end it by the PID its status names (its Job Object takes its
    children with it)."""
    loop = Loop(a.root)
    (loop.root / "STOP").write_text("stop\n", encoding="utf-8")
    st = read_json(STATUS_DIR / "status", {}) or {}
    pid = st.get("pid")
    end = time.time() + a.wait_s
    from tools.lab import winsys  # noqa: PLC0415

    while pid and time.time() < end and winsys.parent_alive(int(pid)):
        time.sleep(5)
    if pid and winsys.parent_alive(int(pid)):
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, check=False)
        print(f"killed {pid}")
    else:
        print("stopped")
    return 0


def cmd_status(a) -> int:
    loop = Loop(a.root)
    st = read_json(STATUS_DIR / "status", {})
    seg = fg.read_jsonl(loop.games / "segments.jsonl")
    idx = fg.read_jsonl(loop.root / "buffer" / "index.jsonl")
    print(json.dumps({"status": st, "segments": len(seg), "fights": sum(s["fights"] for s in seg),
                      "samples": sum(e["rows"] for e in idx), "checkpoints": len(list(loop.checkpoints.glob("c*"))),
                      "incumbent": loop.state["incumbent"], "gated": len(loop.state["gated"]), "gpu": gpu_split(loop)}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("starts", "weights", "run", "gate", "stop", "status"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=str(DEFAULT_ROOT))
        p.add_argument("--procs", type=int, default=6)
        p.add_argument("--prior", choices=("server", "cpu"), default="server")
        if name == "starts":
            p.add_argument("--runs-per-night", type=int, default=50)
        if name in ("run", "gate"):
            p.add_argument("--player-config", default="a3-v1", help="the search config the gate scores (the judge's once it lands)")
            p.add_argument("--gate-k", type=int, default=32)
            p.add_argument("--judge", default="", help="the judge checkpoint a judge config's leaves read (once the judge lands)")
        if name == "gate":
            p.add_argument("--candidate", required=True)
        if name == "stop":
            p.add_argument("--wait-s", type=float, default=300.0)
        if name == "run":
            p.add_argument("--teacher", default="x16", help="the generation's search: the stronger search grades the fights")
            p.add_argument("--k", type=int, default=8, help="fights a lockstep process holds during generation")
            p.add_argument("--segment-min", type=float, default=20.0)
            p.add_argument("--yield-s", type=float, default=150.0)
            p.add_argument("--exact-wall-s", type=float, default=0.25)
            p.add_argument("--checkpoint-min", type=float, default=30.0)
            p.add_argument("--max-epochs", type=float, default=4.0)
            p.add_argument("--train-window", type=int, default=300_000)
            p.add_argument("--buffer-rows", type=int, default=3_000_000)
            p.add_argument("--buffer-days", type=float, default=21.0)
            p.add_argument("--buffer-gb", type=float, default=20.0)
            p.add_argument("--runs-per-night", type=int, default=50)
            p.add_argument("--board", choices=("commit", "dry", "off"), default="commit")
            p.add_argument("--until", default="", help="stop at this UTC stamp (the dry run)")
            p.add_argument("--no-trainer", action="store_true")
            p.add_argument("--no-gate", action="store_true")
    a = ap.parse_args(argv)
    if a.cmd == "starts":
        loop = Loop(a.root)
        print(json.dumps(build_starts(loop, runs_per_night=a.runs_per_night, procs=a.procs), indent=1)[:2000])
        return 0
    if a.cmd == "weights":
        print(json.dumps(kind_weights(Loop(a.root), procs=a.procs, prior=a.prior), indent=1))
        return 0
    if a.cmd == "gate":
        loop = Loop(a.root)
        d = Path(a.candidate)
        print(json.dumps(run_gate(loop, d.name, d, a.player_config, a.procs, a.gate_k, a.prior, judge=a.judge or None), indent=1))
        return 0
    if a.cmd == "stop":
        return cmd_stop(a)
    if a.cmd == "status":
        return cmd_status(a)
    return Daemon(a).run()


if __name__ == "__main__":
    raise SystemExit(main())
