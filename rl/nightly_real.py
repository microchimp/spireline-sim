#!/usr/bin/env python3
"""The nightly real-game block (D33 addendum 2 item 1): the lab job `nightly-real` runs this driver.

    python rl/nightly_real.py --lab LAB --agent-root AGENT --game GAME --lib DIR [--runs 2000] [--runs-promoted N] [--procs 12] [--recycle 60]

For the incumbent (agent-root/current.json) and each generation promoted since the last nightly (promotions.tsv rows written after
LAB/nightly/last-ok whose decision is `promoted`), one fresh block of seeds (gen0.e_block rule, name N-<date> for the incumbent and
N-<date>-<agent> for the others, written to LAB/seeds/, checked disjoint from every known block, E and S blocks included) is played
by `gen0.py score` in lockstep with the emulator arm on `--engine core2`. The block's folder is LAB/nightly/<date>/<agent>/ (the
real game's runs.jsonl, divergences.jsonl, summary.json, compare.json; the emulator arm is its sibling <agent>-emu/), ranked in
place by tools/lab/rank.py (ranked.json, top20.md, by_family.*). A block a stopped job left unfinished is resumed in place (`gen0.py
score --resume`): its folder is not cleared and only the seeds without a row are played; a block with the finished mark is skipped.

Runs per agent (owner, 2026-10-30): `--runs-promoted` on a night that scores a generation promoted since the last nightly, else
`--runs` (the lab's jobs.json: 2,000 on a promotion day, 500 otherwise). Without `--runs-promoted` every night plays `--runs`.

GPU_OK: `gen0.py score --gpu-ok-tag lab-nightly` keeps the line `lab-nightly` first in the GPU_OK file while the real-game arm plays
(the GPU lane may fit beside it), never during the emulator arm; every start of this driver first removes any such line.

summary.json keeps the envproto summary and gains `nightly`: the numbers counted with a crashed run as a loss. That is
gen0.real_runs(): a crash row becomes {win: False, floor: 0, crash: True}, which gen0.engine_block counts as a run with floor 0 and
no win, and gen_loop.boss_kill never as a boss kill.

The job takes LAB's neighbour QUIET.lock with the line `lab-nightly <UTC>` before the first game process, rewrites the line while it
runs (so another's 2 h staleness rule never reads a live block as stale), removes it after the last process, and waits up to 2 h while
another's line is under 2 h old; after that it writes {"skipped": ...} to result.json and stops (the lab tries again later).
The lab's PAUSE is never touched here. Nothing here starts a game directly: `gen0.py score` does, through referee/envproto.py.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import threading
import time
from collections.abc import Callable
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TAG = "lab-nightly"
STALE_H = 2.0  # another's lock line under this old is live
RESUME_NIGHT_H = 36.0  # an unfinished night whose newest row is under this old is resumed under its own date
UTC_FORMAT = "%Y-%m-%dT%H:%M:%SZ"


def utc_now(t: float | None = None) -> str:
    return dt.datetime.fromtimestamp(time.time() if t is None else t, dt.UTC).strftime(UTC_FORMAT)


def parse_utc(s: str) -> float | None:
    try:
        return dt.datetime.strptime(s, UTC_FORMAT).replace(tzinfo=dt.UTC).timestamp()
    except ValueError:
        return None


# ---------------------------------------------------------------- the QUIET.lock protocol

class QuietLock:
    """`<tag> <UTC>` in one file. Taken when absent, ours (same tag: a resumed job), or its line is older than `stale_h`; otherwise
    waited for, up to `wait_h`. While held, `heartbeat` rewrites the line so the holder never reads as stale."""

    def __init__(self, path: Path, tag: str = TAG, stale_h: float = STALE_H, wait_h: float = 2.0, poll_s: float = 60.0,
                 clock: Callable[[], float] = time.time, sleep: Callable[[float], None] = time.sleep):
        self.path, self.tag, self.stale_h, self.wait_h, self.poll_s = Path(path), tag, stale_h, wait_h, poll_s
        self.clock, self.sleep = clock, sleep
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def holder(self) -> tuple[str, float] | None:
        """(tag, age in hours) of the line in the file; None when there is none. The age is the line's UTC, else the file's mtime."""
        try:
            line = self.path.read_text(encoding="utf-8", errors="replace").strip().splitlines()[0]
        except (OSError, IndexError):
            return None
        parts = line.split()
        t = parse_utc(parts[1]) if len(parts) > 1 else None
        if t is None:
            with contextlib.suppress(OSError):
                t = self.path.stat().st_mtime
        return (parts[0] if parts else "", (self.clock() - t) / 3600 if t is not None else 0.0)

    def blocked_by(self) -> str | None:
        h = self.holder()
        if h is None or h[0] == self.tag or h[1] >= self.stale_h:
            return None
        return f"{h[0] or 'a lock'} ({h[1]:.1f} h old)"

    def _write(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(f"{self.tag} {utc_now(self.clock())} pid={os.getpid()}\n", encoding="utf-8")  # the runner clears a line whose pid is gone

    def acquire(self) -> bool:
        deadline = self.clock() + self.wait_h * 3600
        while True:
            if self.blocked_by() is None:
                self._write()
                return True
            if self.clock() >= deadline:
                return False
            self.sleep(self.poll_s)

    def start_heartbeat(self, every_s: float = 900.0) -> None:
        def beat() -> None:
            while not self._stop.wait(every_s):
                h = self.holder()
                if h is not None and h[0] == self.tag:
                    with contextlib.suppress(OSError):
                        self._write()

        self._thread = threading.Thread(target=beat, daemon=True)
        self._thread.start()

    def release(self) -> None:
        self._stop.set()
        h = self.holder()
        if h is not None and h[0] == self.tag:  # never remove another's line
            self.path.unlink(missing_ok=True)


# ---------------------------------------------------------------- who is scored

def promoted_dir(agent_root: Path, row: dict[str, str]) -> Path | None:
    """The directory a promotions.tsv row names, or None when there is none. `gen_loop.promote` writes the candidate folder's
    name (`candidate`, `candidate-r1`) in `cand_dir`, which lives under AGENT/gen<cand_gen>/; a full path is taken as it is."""
    d = row.get("cand_dir") or ""
    if not d:
        return None
    p = Path(d)
    if not p.is_absolute():
        try:
            p = Path(agent_root) / f"gen{int(row['cand_gen'])}" / d
        except (KeyError, ValueError):
            return None
    return p if p.is_dir() else None


def agents_to_score(agent_root: Path, since: float | None) -> list[dict[str, Any]]:
    """The incumbent first, then each generation promoted since `since` (epoch seconds; None: only the incumbent), once per directory.
    Each: label (the folder name), generation, dir, role. An incumbent that was itself promoted since `since` (the usual promotion
    day: the new generation is the incumbent) keeps the role `incumbent` and gets `promoted_since` True."""
    import gen_loop  # noqa: PLC0415

    cur = gen_loop.read_json(Path(agent_root) / "current.json")
    out = [{"label": f"g{int(cur['generation'])}", "generation": int(cur["generation"]), "dir": str(cur["dir"]), "role": "incumbent"}]
    seen = {Path(out[0]["dir"]).resolve()}
    for row in gen_loop.read_promotions(Path(agent_root) / "promotions.tsv") if since is not None else []:
        if row.get("decision") != "promoted":
            continue
        t = parse_utc(row.get("written", ""))
        d = promoted_dir(agent_root, row)
        if t is None or t <= since or d is None:
            continue
        if d.resolve() == Path(out[0]["dir"]).resolve():
            out[0]["promoted_since"] = True
        if d.resolve() in seen:
            continue
        seen.add(d.resolve())
        label = f"g{int(row['cand_gen'])}"
        if any(x["label"] == label for x in out):  # a retrain of the same generation: its directory name tells them apart
            label += "-" + d.name
        out.append({"label": label, "generation": int(row["cand_gen"]), "dir": str(d), "role": "promoted"})
    return out


def promoted_labels(agents: list[dict[str, Any]]) -> list[str]:
    """The agents tonight that were promoted since the last nightly: the `promoted` ones and an incumbent promoted since."""
    return [ag["label"] for ag in agents if ag.get("role") == "promoted" or ag.get("promoted_since")]


def night_runs(agents: list[dict[str, Any]], runs: int, runs_promoted: int | None) -> int:
    """Runs per agent tonight: `runs_promoted` when `agents` holds a generation promoted since the last nightly (`promoted_labels`:
    the incumbent itself counts when it was promoted since), else `runs`."""
    if runs_promoted and promoted_labels(agents):
        return runs_promoted
    return runs


def seed_block_name(date: str, agent: dict[str, Any]) -> str:
    return f"N-{date}" if agent["role"] == "incumbent" else f"N-{date}-{agent['label']}"


def make_block(lab: Path, name: str, n: int, extra_seed_dirs: list[Path]) -> tuple[Path, list[str]]:
    """Write the block's seed file under LAB/seeds/ and check it against every other known block (E, S, N, the held-out seeds and the
    lab's self-play runs). Raises ValueError on a shared seed or a name that would read as an E or S block."""
    import gen0  # noqa: PLC0415
    import gen_loop  # noqa: PLC0415

    if not name.startswith("N-"):
        raise ValueError(f"nightly block {name!r}: not an N- name")
    seeds_dir = gen_loop.lab_seed_dir(lab)
    seeds = gen0.e_block(name, n)
    others = gen_loop.known_seed_sets(seeds_dir, [Path(lab) / "runs"], name, extra_seed_dirs)
    gen_loop.check_disjoint(seeds, others)
    gen0.write_seed_file(name, n, seeds_dir / f"{name}.json")
    return seeds_dir / f"{name}.json", seeds


def unfinished_night(lab: Path, now: float | None = None, max_age_h: float = RESUME_NIGHT_H) -> str | None:
    """The date of the newest LAB/nightly/<date> folder that holds a started agent block without its finished mark (the agent's
    summary.json with `nightly`): an emulator arm (`<agent>-emu/runs.jsonl`) or a real folder that a stopped job left behind,
    when its newest runs.jsonl is under `max_age_h` old. None: every night is finished or too old, so a fresh one starts.

    The runner supersedes a stopped job once main has moved, and the next nightly job would start a fresh night under today's
    date, leaving the old emulator arm's rows orphaned; resuming the old date instead keeps them (gen0.run_block resume) and
    plays the real arm the night was for."""
    root = Path(lab) / "nightly"
    t = time.time() if now is None else now
    try:
        dates = sorted((p for p in root.iterdir() if p.is_dir()), key=lambda p: p.name, reverse=True)
    except OSError:
        return None
    for d in dates:
        try:
            dt.datetime.strptime(d.name, "%Y-%m-%d")
        except ValueError:
            continue
        labels = {p.name[:-4] if p.name.endswith("-emu") else p.name for p in d.iterdir() if p.is_dir()}
        newest = max((f.stat().st_mtime for f in d.glob("*/runs.jsonl")), default=None)
        if not labels or newest is None or t - newest > max_age_h * 3600:
            continue
        for label in labels:
            done = d / label / "summary.json"
            try:
                finished = "nightly" in json.loads(done.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                finished = False
            if not finished:
                return d.name
    return None


# ---------------------------------------------------------------- one agent's block

def nightly_numbers(folder: Path, date: str, agent: dict[str, Any], seed_block: str, planned: int, engine: str) -> dict[str, Any]:
    """The block's numbers with a crashed run counted as a loss (gen0.real_runs: floor 0, no win; boss_kill: never)."""
    import gen0  # noqa: PLC0415
    import gen_loop  # noqa: PLC0415

    rows = gen0.real_runs(folder / "runs.jsonl") if (folder / "runs.jsonl").is_file() else []
    t = gen0.engine_block(rows) if rows else {"runs": 0, "wins": 0, "win_rate": None, "floor": {"mean": None}, "crashes": 0}
    summ = json.loads((folder / "summary.json").read_text(encoding="utf-8")) if (folder / "summary.json").is_file() else {}
    boss = sum(bool(gen_loop.boss_kill(r)) for r in rows)
    mean = t["floor"]["mean"]
    from tools.lab import rank  # noqa: PLC0415

    stalls = rank.stalls_of(folder / "runs.jsonl", emu_jsonl=folder.parent / (folder.name + "-emu") / "runs.jsonl")  # the real arm: timeouts and
    # runs that answered one screen the same way over 50 times; the sibling emulator arm: its timeouts (the wall cap), L2's run-layer loops
    return {"date": date, "agent": agent["label"], "generation": agent["generation"], "role": agent["role"], "seed_block": seed_block,
            "engine": engine, "runs_planned": planned, "runs": t["runs"], "wins": t["wins"], "win_rate": t["win_rate"],
            "mean_floor": None if mean is None else round(mean, 3), "boss_kills": boss,
            "boss_kill_rate": boss / t["runs"] if t["runs"] else None, "crashes": t["crashes"],
            "stalls": stalls, "divergence_rate": summ.get("divergence_rate"), "diverged_decisions": summ.get("diverged_decisions"),
            "reanchors": summ.get("reanchors"), "reanchor_failed": summ.get("reanchor_failed"), "wall_s": summ.get("wall_s"),
            "counted": "a crashed run is a loss: floor 0, no win, no boss kill"}


def gpu_ok_default() -> str:
    import gen0  # noqa: PLC0415

    return gen0.GPU_OK_FILE


def play_agent(a: argparse.Namespace, agent: dict[str, Any], date: str, date_dir: Path) -> dict[str, Any]:
    from tools.lab import rank  # noqa: PLC0415

    folder = date_dir / agent["label"]
    emu = date_dir / (agent["label"] + "-emu")
    name = seed_block_name(date, agent)
    done = folder / "summary.json"
    if done.is_file():  # the job was stopped and resumed: a finished block is not played twice
        with contextlib.suppress(OSError, ValueError):
            if "nightly" in json.loads(done.read_text(encoding="utf-8")):
                return json.loads(done.read_text(encoding="utf-8"))["nightly"]
    seeds_file, seeds = make_block(Path(a.lab), name, a.runs, [Path(p) for p in a.seeds_also])
    cmd = [sys.executable, str(ROOT / "rl" / "gen0.py"), "score", "--game", a.game, "--seeds-file", str(seeds_file), "--runs", str(a.runs),
           "--procs", str(a.procs), "--config", a.config, "--stop-after-h", str(a.stop_after_h), "--lib", a.lib,
           "--gen-dir", agent["dir"], "--out", str(folder), "--engine", a.engine, "--recycle", str(a.recycle), "--resume",
           # the GPU lane may fit beside the real-game arm (not the emulator arm, which plays through the prior server on the GPU)
           "--gpu-ok-tag", TAG, "--gpu-ok-file", str(getattr(a, "gpu_ok_file", None) or gpu_ok_default())]
    if getattr(a, "sample_machine", 0):  # the one-night sampler of the slow windows: the real arm only (`gen0.py score` hands it to envproto play)
        cmd += ["--sample-machine", f"{a.sample_machine:g}"]
    # a stopped, unfinished block is resumed in its folder, real arm and emulator arm alike: the seeds that already have a row are
    # kept (envproto play --resume and gen0.run_block resume; each arm's identity file keeps rows of another agent or setting out),
    # so the lab stopping a 7-9 hour block on the owner's input, on load or on a landing costs only the runs in flight
    print(f"[{utc_now()}] {agent['label']} ({agent['role']}, generation {agent['generation']}): {name}, {len(seeds)} seeds", flush=True)
    rc = subprocess.run(cmd, check=False).returncode
    shutil.rmtree(folder / "scratch", ignore_errors=True)  # the game processes' working copies
    nums = nightly_numbers(folder, date, agent, name, len(seeds), a.engine)
    nums["rc"] = rc
    if (folder / "summary.json").is_file():
        doc = json.loads((folder / "summary.json").read_text(encoding="utf-8"))
        (folder / "summary.json").write_text(json.dumps({**doc, "nightly": nums}, indent=1), encoding="utf-8")
    if (folder / "divergences.jsonl").is_file():
        with contextlib.suppress(OSError, ValueError, KeyError):
            rank.write(folder, folder, emu_dir=emu if (emu / "runs.jsonl").is_file() else None)
    return nums


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--lab", required=True, help="the lab folder: LAB/nightly/ is written, LAB/seeds/ gets the blocks")
    ap.add_argument("--agent-root", required=True)
    ap.add_argument("--game", required=True)
    ap.add_argument("--lib", required=True, help="directory of the native library")
    ap.add_argument("--out", help="the job folder: result.json goes here (the runner's {job})")
    ap.add_argument("--runs", type=int, default=2000)
    ap.add_argument("--runs-promoted", type=int, default=None,
                    help="runs per agent on a night that scores a generation promoted since the last nightly (default: --runs)")
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--recycle", type=int, default=60)
    ap.add_argument("--config", default="a3-v1")
    ap.add_argument("--engine", choices=("old", "core2"), default="core2")
    ap.add_argument("--stop-after-h", type=float, default=4.0, help="per agent: no new run after this many hours")
    ap.add_argument("--date", help="the night's folder and block name (default: the newest unfinished night under 36 h old, else today)")
    ap.add_argument("--quiet-lock", default="D:/spireline/referee/QUIET.lock", help="the shared QUIET.lock file")
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--gpu-ok-file", default=None, help="the GPU_OK file (default: gen0.GPU_OK_FILE): `gen0.py score` keeps the line "
                    "lab-nightly in it while the real-game arm plays; a stale one is removed at every start")
    ap.add_argument("--sample-machine", type=float, default=30.0, help="seconds between the lines of machine.jsonl in each real-game block's "
                    "folder (referee/machine_sampler.py; the real arm only); 0: off")
    ap.add_argument("--wait-h", type=float, default=2.0)
    ap.add_argument("--no-lock", action="store_true", help="the caller holds the QUIET.lock (the dry run)")
    ap.add_argument("--seeds-also", action="append", default=[], help="another folder of seed blocks to stay disjoint from (read only)")
    ap.add_argument("--since", help="UTC time (default: LAB/nightly/last-ok's): promotions after it are scored too")
    a = ap.parse_args(argv)
    import gen0  # noqa: PLC0415

    a.gpu_ok_file = a.gpu_ok_file or gen0.GPU_OK_FILE
    gen0.gpu_ok_remove(a.gpu_ok_file, TAG)  # first of all: a line a killed job left must not cover this start's emulator arm
    lab = Path(a.lab)
    if a.date is None:
        a.date = unfinished_night(lab)
        if a.date is not None:
            print(f"[{utc_now()}] resuming the unfinished night {a.date} (both arms of each block keep their rows)", flush=True)
        else:
            a.date = time.strftime("%Y-%m-%d")
    out = Path(a.out) if a.out else lab / "nightly"
    out.mkdir(parents=True, exist_ok=True)
    lock = QuietLock(Path(a.quiet_lock), a.tag, wait_h=a.wait_h)
    if not a.no_lock:
        if not lock.acquire():
            why = f"QUIET.lock stayed taken by {lock.blocked_by()} for {a.wait_h:g} h"
            print(f"[{utc_now()}] skipped: {why}", flush=True)
            (out / "result.json").write_text(json.dumps({"skipped": why, "agents": {}}), encoding="utf-8")
            return 0
        lock.start_heartbeat()
    try:
        since = parse_utc(a.since) if a.since else None
        ok = lab / "nightly" / "last-ok"
        if since is None and ok.is_file():
            since = ok.stat().st_mtime
        t0 = time.time()
        date_dir = lab / "nightly" / a.date
        agents: dict[str, Any] = {}
        todo = agents_to_score(Path(a.agent_root), since)
        a.runs = night_runs(todo, a.runs, a.runs_promoted)
        promoted = promoted_labels(todo)
        print(f"[{utc_now()}] {a.runs} runs per agent ({'promoted since the last nightly: ' + ', '.join(promoted) if promoted else 'no promotion since the last nightly'})", flush=True)
        for ag in todo:
            try:
                agents[ag["label"]] = play_agent(a, ag, a.date, date_dir)
            except (OSError, ValueError, KeyError) as e:  # one agent's problem (a shared seed, a missing directory) does not stop the others
                agents[ag["label"]] = {"date": a.date, "agent": ag["label"], "error": f"{type(e).__name__}: {e}"[:300]}
                print(f"[{utc_now()}] {ag['label']}: {agents[ag['label']]['error']}", flush=True)
        good = [v for v in agents.values() if "error" not in v]
        inc = good[0] if good else {}
        res = {"date": a.date, "engine": a.engine, "runs_per_agent": a.runs, "agents": agents, "wall_s": round(time.time() - t0, 1),
               # the yield columns (tools/lab/mix.py) are the first agent's
               "runs": inc.get("runs"), "wins": inc.get("wins"), "mean_floor": inc.get("mean_floor"), "crashes": inc.get("crashes"),
               "divergences": inc.get("diverged_decisions")}
        (out / "result.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
        return 0 if good else 1
    finally:
        if not a.no_lock:
            lock.release()


if __name__ == "__main__":
    sys.exit(main())
