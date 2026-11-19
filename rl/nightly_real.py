#!/usr/bin/env python3
"""The nightly real-game block (D33 addendum 2 item 1): the lab job `nightly-real` runs this driver.

    python rl/nightly_real.py --lab LAB --agent-root AGENT --game GAME --lib DIR [--runs 2000] [--runs-promoted N] [--emu-runs N] [--procs 12]
                              [--recycle 60]

Two triggers (2026-11-06: the real-game check is the bottleneck, so it runs when the orchestrator asks for it after a promotion, not
on a clock). A *promotion* block runs when LAB/nightly/pending exists (written by hand; no code writes it): it plays the incumbent when
it was promoted since the last block, at `--runs-promoted` (2000), plus the copy arm's `--emu-runs`, and does not play the incumbent's
`--runs` block (an incumbent not promoted since is not played: the block is idle). A *clock* block (LAB/nightly/last-ok older than
the job's `every_h`, the weekly real-game check: 168 hours makes it weekly) plays the incumbent alone: at `--runs-promoted` when a
generation was promoted since the last block, else at `--runs`. `--trigger` forces one (default: promotion when the marker exists at the start).
Neither block plays the generations promoted between blocks (the lab promoted five in the 13 days to 2026-11-19: five blocks of 7.6 hours
for five players the next promotion replaced): only `--all-promoted` brings that back (a promotion block then plays each generation
promoted since, a clock block the incumbent and each of them, all at `--runs-promoted`); the result and the log say which were left out.
The driver takes its list at its start time (`started_epoch` in result.json); the lab stamps last-ok with that time and removes the
marker when the block ends, unless it was written after that time, so a promotion that lands while a block plays is the next block's.

For the incumbent (agent-root/current.json) and, with `--all-promoted`, each generation promoted since the last block (promotions.tsv rows
written after LAB/nightly/last-ok whose decision is `promoted`), one fresh block of seeds (gen0.e_block rule, name N-<date> for the incumbent and
N-<date>-<agent> for the others, written to LAB/seeds/, checked disjoint from every known block, E and S blocks included) is played
by `gen0.py score` in lockstep with the emulator arm on `--engine core2`. The block's folder is LAB/nightly/<block>/<agent>/, <block> the
UTC start `YYYY-MM-DDTHHMM` (two blocks on one day do not collide; the folders of 2026-11-06 and earlier are named by date alone, and every
reader takes both; the seed blocks are `N-<block>[-<agent>]`) (the
real game's runs.jsonl, divergences.jsonl, summary.json, compare.json; the emulator arm is its sibling <agent>-emu/), ranked in
place by tools/lab/rank.py (ranked.json, top20.md, by_family.*). A block a stopped job left unfinished is resumed in place (`gen0.py
score --resume`): its folder is not cleared and only the seeds without a row are played; a block with the finished mark is skipped.

Runs per agent (owner, 2026-10-30): `--runs-promoted` on a night that scores a generation promoted since the last nightly, else
`--runs` (the lab's jobs.json: 2,000 on a promotion day, 500 otherwise). Without `--runs-promoted` every night plays `--runs`.

Copy arm on a sample (`--emu-runs N`, 2026-11-04): the emulator arm plays only the first N seeds of each block (a fixed prefix of the
block's seed file, so the set both arms share is the same every night and on every resume; `gen0.emu_arm_seeds`); the real arm still
plays them all. compare.json, the rank join (`rank.write`) and the readers that pair the arms work on the seeds both arms hold, and
`nightly` carries `emu_runs_planned` (the copy arm's seeds), `emu_rows` (rows it holds) and `paired_seeds` (seeds with a row in both
arms). A block a stopped job left unfinished resumes both arms with the same prefix. Default: every seed of the block.

GPU_OK: `gen0.py score --gpu-ok-tag lab-nightly` keeps the line `lab-nightly` first in the GPU_OK file while the real-game arm plays
(the GPU lane may fit beside it), never during the emulator arm; every start of this driver first removes any such line.

summary.json keeps the envproto summary and gains `nightly`: the numbers counted with a crashed run as a loss. That is
gen0.real_runs(): a crash row becomes {win: False, floor: 0, crash: True}, which gen0.engine_block counts as a run with floor 0 and
no win, and gen_loop.boss_kill never as a boss kill. A game process that does not come up is not a crash row at once: referee/envproto.py
kills it and starts another on a fresh folder for 20 minutes (`--boot-budget-s`); `nightly` carries `boot_retries` (starts retried) and
`boot_gave_up` (seeds whose every try failed, a crash row each), and the block folder's boot.jsonl has a line for each failed start with the
process's CPU seconds and bytes read when it was given up (what a stalled start was waiting on).

The job takes LAB's neighbour QUIET.lock with the line `lab-nightly <UTC>` before the first game process, rewrites the line while it
runs (so another's 2 h staleness rule never reads a live block as stale), removes it after the last process, and waits up to 2 h while
another's line is under 2 h old; after that it writes {"skipped": ...} to result.json and stops (the lab tries again later).
The lock code is rl/quiet_lock.py, shared with `gen0.py score --quiet-lock` (tag `lab-score`, the daytime real-game score).
The lab's PAUSE is never touched here. Nothing here starts a game directly: `gen0.py score` does, through referee/envproto.py.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import json
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT), str(ROOT / "rl")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

TAG = "lab-nightly"
RESUME_NIGHT_H = 36.0  # an unfinished night whose newest row is under this old is resumed under its own date
BLOCK_FORMAT = "%Y-%m-%dT%H%M"  # the folder of a block started now (UTC); folders up to 2026-11-06 are `%Y-%m-%d`
BLOCK_RE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:T\d{4})?$")
TRIGGERS = ("auto", "promotion", "clock")

import quiet_lock  # noqa: E402 -- rl/ is on sys.path (above)
from quiet_lock import STALE_H, UTC_FORMAT, parse_utc, utc_now  # noqa: E402,F401 -- the QUIET.lock protocol moved to rl/quiet_lock.py (shared with `gen0.py score`)


class QuietLock(quiet_lock.QuietLock):
    """rl/quiet_lock.QuietLock with this driver's tag as the default."""

    def __init__(self, path: Path, tag: str = TAG, **kw: Any):
        super().__init__(path, tag, **kw)


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


def block_name(now: float | None = None) -> str:
    """The folder (and block label) of a block started at `now` (epoch seconds, default now): the UTC date and minute, so two blocks of one
    day have folders of their own."""
    return time.strftime(BLOCK_FORMAT, time.gmtime(time.time() if now is None else now))


def is_block_folder(name: str) -> bool:
    """A LAB/nightly/<block> folder name: a date (the blocks up to 2026-11-06) or a date and start time (`2026-11-06T0312`)."""
    return bool(BLOCK_RE.match(name))


def pending_marker(lab: Path) -> Path:
    return Path(lab) / "nightly" / "pending"


def trigger_of(lab: Path, requested: str = "auto") -> str:
    """`promotion` when the lab's pending marker exists (or `requested` says so), else `clock`."""
    if requested in ("promotion", "clock"):
        return requested
    return "promotion" if pending_marker(lab).is_file() else "clock"


def promotion_agents(agents: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """The agents a promotion block plays: those promoted since the last block (`promoted_labels`); the incumbent only when it is one of
    them. An incumbent that was not promoted since is not played at `--runs` by a promotion block (that is the clock block's job)."""
    keep = set(promoted_labels(agents))
    return [ag for ag in agents if ag["label"] in keep]


def block_agents(agents: list[dict[str, Any]], trigger: str, all_promoted: bool = False) -> list[dict[str, Any]]:
    """The agents a block plays out of `agents_to_score`'s list (the incumbent first). By default the incumbent alone: a clock block always, a
    promotion block only when the incumbent was itself promoted since the last block (else nothing: the block is idle). `all_promoted` plays
    the old list: a promotion block's `promotion_agents`, a clock block's every agent."""
    if all_promoted:
        return promotion_agents(agents) if trigger == "promotion" else list(agents)
    inc = [ag for ag in agents if ag.get("role") == "incumbent"][:1]
    if trigger == "promotion":
        return [ag for ag in inc if ag.get("promoted_since")]
    return inc


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
    """The name of the newest LAB/nightly/<block> folder (a date, or a date and start time) that holds a started agent block without its finished mark (the agent's
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
        if not is_block_folder(d.name):
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

def emu_runs_tonight(emu_runs: int | None, runs: int) -> int | None:
    """The copy arm's seed count for a block of `runs` seeds: `emu_runs` cut to the block (None: every seed of the block)."""
    return None if emu_runs is None else min(int(emu_runs), int(runs))


def seeds_with_rows(runs_jsonl: Path) -> set[str]:
    """The seeds that have a row in a runs.jsonl of either arm (the real arm's end and crash events, the copy arm's rows); a half-written
    last line is skipped."""
    out: set[str] = set()
    try:
        lines = Path(runs_jsonl).read_text(encoding="utf-8", errors="replace").splitlines()
    except OSError:
        return out
    for line in lines:
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if isinstance(r, dict) and r.get("seed") is not None and r.get("ev") in (None, "end", "crash"):
            out.add(str(r["seed"]))
    return out


def nightly_numbers(folder: Path, date: str, agent: dict[str, Any], seed_block: str, planned: int, engine: str,
                    emu_planned: int | None = None) -> dict[str, Any]:
    """The block's numbers with a crashed run counted as a loss (gen0.real_runs: floor 0, no win; boss_kill: never). `emu_planned` is the
    seeds the copy arm was to play (`--emu-runs`; None: all of `planned`)."""
    import gen0  # noqa: PLC0415
    import gen_loop  # noqa: PLC0415

    rows = gen0.real_runs(folder / "runs.jsonl") if (folder / "runs.jsonl").is_file() else []
    t = gen0.engine_block(rows) if rows else {"runs": 0, "wins": 0, "win_rate": None, "floor": {"mean": None}, "crashes": 0}
    summ = json.loads((folder / "summary.json").read_text(encoding="utf-8")) if (folder / "summary.json").is_file() else {}
    boss = sum(bool(gen_loop.boss_kill(r)) for r in rows)
    mean = t["floor"]["mean"]
    from tools.lab import rank  # noqa: PLC0415

    emu_jsonl = folder.parent / (folder.name + "-emu") / "runs.jsonl"
    emu_seeds = seeds_with_rows(emu_jsonl)  # the copy arm may hold fewer seeds than the real one (--emu-runs): pairs are the seeds in both
    paired_seeds = len(emu_seeds & seeds_with_rows(folder / "runs.jsonl"))
    stalls = rank.stalls_of(folder / "runs.jsonl", emu_jsonl=emu_jsonl)  # the real arm: timeouts and
    # runs that answered one screen the same way over 50 times; the sibling emulator arm: its timeouts (the wall cap), L2's run-layer loops
    return {"date": date, "agent": agent["label"], "generation": agent["generation"], "role": agent["role"], "seed_block": seed_block,
            "engine": engine, "runs_planned": planned, "emu_runs_planned": planned if emu_planned is None else emu_planned,
            "emu_rows": len(emu_seeds), "paired_seeds": paired_seeds, "runs": t["runs"], "wins": t["wins"], "win_rate": t["win_rate"],
            "mean_floor": None if mean is None else round(mean, 3), "boss_kills": boss,
            "boss_kill_rate": boss / t["runs"] if t["runs"] else None, "crashes": t["crashes"],
            "stalls": stalls, "divergence_rate": summ.get("divergence_rate"), "diverged_decisions": summ.get("diverged_decisions"),
            "reanchors": summ.get("reanchors"), "reanchor_failed": summ.get("reanchor_failed"), "wall_s": summ.get("wall_s"),
            # the second, unranked comparison (referee/hidden_diff.py): divergence records that carry a difference in a field the
            # ranking does not look at, and the decisions with one; None: a summary from before it. Not in the ranked groups.
            "hidden_records": (summ.get("hidden") or {}).get("records"),
            "hidden_diverged_decisions": (summ.get("hidden") or {}).get("diverged_decisions"),
            # game processes that did not come up: starts retried on a fresh process, and seeds whose tries all failed (a crash row each);
            # the lines are in the block folder's boot.jsonl (referee/envproto.py boot_game). None: a summary from before the retry
            "boot_retries": summ.get("boot_retries"), "boot_gave_up": summ.get("boot_gave_up"),
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
    emu_n = emu_runs_tonight(getattr(a, "emu_runs", None), a.runs)
    if emu_n is not None:  # the copy arm plays this prefix of the block's seeds; a resumed block gets the same one
        cmd += ["--emu-runs", str(emu_n)]
    if getattr(a, "sample_machine", 0):  # the one-night sampler of the slow windows: the real arm only (`gen0.py score` hands it to envproto play)
        cmd += ["--sample-machine", f"{a.sample_machine:g}"]
    # a stopped, unfinished block is resumed in its folder, real arm and emulator arm alike: the seeds that already have a row are
    # kept (envproto play --resume and gen0.run_block resume; each arm's identity file keeps rows of another agent or setting out),
    # so the lab stopping a 7-9 hour block on the owner's input, on load or on a landing costs only the runs in flight
    print(f"[{utc_now()}] {agent['label']} ({agent['role']}, generation {agent['generation']}): {name}, {len(seeds)} seeds", flush=True)
    rc = subprocess.run(cmd, check=False).returncode
    shutil.rmtree(folder / "scratch", ignore_errors=True)  # the game processes' working copies
    nums = nightly_numbers(folder, date, agent, name, len(seeds), a.engine, emu_n)
    nums["rc"] = rc
    nums["trigger"] = getattr(a, "trigger_used", None)
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
    ap.add_argument("--emu-runs", type=int, default=None,
                    help="the emulator (copy) arm plays only the first N seeds of each block, the real arm all of them (default: all); "
                    "the pairing and the readers use the seeds both arms hold")
    ap.add_argument("--procs", type=int, default=12)
    ap.add_argument("--recycle", type=int, default=60)
    ap.add_argument("--config", default="a3-v1")
    ap.add_argument("--engine", choices=("old", "core2"), default="core2")
    ap.add_argument("--stop-after-h", type=float, default=4.0, help="per agent: no new run after this many hours")
    ap.add_argument("--date", help="the block's folder and seed-block label (default: the newest unfinished block under 36 h old, else the "
                    "UTC start time, `YYYY-MM-DDTHHMM`)")
    ap.add_argument("--trigger", choices=TRIGGERS, default="auto",
                    help="promotion: play the generations promoted since the last block (the incumbent only if among them); clock: the "
                    "incumbent at --runs as well; auto: promotion when LAB/nightly/pending exists at the start")
    ap.add_argument("--quiet-lock", default="D:/spireline/referee/QUIET.lock", help="the shared QUIET.lock file")
    ap.add_argument("--tag", default=TAG)
    ap.add_argument("--gpu-ok-file", default=None, help="the GPU_OK file (default: gen0.GPU_OK_FILE): `gen0.py score` keeps the line "
                    "lab-nightly in it while the real-game arm plays; a stale one is removed at every start")
    ap.add_argument("--sample-machine", type=float, default=30.0, help="seconds between the lines of machine.jsonl in each real-game block's "
                    "folder (referee/machine_sampler.py; the real arm only); 0: off")
    ap.add_argument("--wait-h", type=float, default=2.0)
    ap.add_argument("--no-lock", action="store_true", help="the caller holds the QUIET.lock (the dry run)")
    ap.add_argument("--seeds-also", action="append", default=[], help="another folder of seed blocks to stay disjoint from (read only)")
    ap.add_argument("--since", help="UTC time (default: LAB/nightly/last-ok's): promotions after it count as promoted since the last block. "
                    "A time older than last-ok's (a block started after it) is read as last-ok's: a --since left in the job's command never turns "
                    "every later block into a promotion block")
    ap.add_argument("--all-promoted", action="store_true", help="play every generation promoted since the last block as well (the pre-2026-11-19 night); "
                    "by default a block plays the incumbent alone")
    a = ap.parse_args(argv)
    if a.emu_runs is not None and a.emu_runs < 1:
        ap.error("--emu-runs must be at least 1")
    import gen0  # noqa: PLC0415

    a.gpu_ok_file = a.gpu_ok_file or gen0.GPU_OK_FILE
    gen0.gpu_ok_remove(a.gpu_ok_file, TAG)  # first of all: a line a killed job left must not cover this start's emulator arm
    lab = Path(a.lab)
    if a.date is None:
        a.date = unfinished_night(lab)
        if a.date is not None:
            print(f"[{utc_now()}] resuming the unfinished night {a.date} (both arms of each block keep their rows)", flush=True)
        else:
            a.date = block_name()
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
        t0 = time.time()  # the block's list is taken now: a promotion after this is the next block's (jobs.finish stamps last-ok with it)
        since = parse_utc(a.since) if a.since else None
        ok = lab / "nightly" / "last-ok"
        last_block = ok.stat().st_mtime if ok.is_file() else None  # the last block's start (jobs.finish stamps it)
        if since is None:
            since = last_block
        elif last_block is not None and last_block > since:  # a --since the last block has passed is spent
            print(f"[{utc_now()}] --since {a.since} is older than the last block's start ({utc_now(last_block)}): the last block's start is used", flush=True)
            since = last_block
        date_dir = lab / "nightly" / a.date
        agents: dict[str, Any] = {}
        a.trigger_used = trigger_of(lab, a.trigger)
        listed = agents_to_score(Path(a.agent_root), since)
        promoted = promoted_labels(listed)  # every generation promoted since the last block (the incumbent among them when it was)
        todo = block_agents(listed, a.trigger_used, bool(getattr(a, "all_promoted", False)))
        played = {ag["label"] for ag in todo}
        left_out = [x for x in promoted if x not in played]  # promoted since, not played (--all-promoted plays them)
        if not todo:  # a promotion block whose promotion the block before already took: nothing to play, nothing stamped
            why = "no generation promoted since the last block" if not promoted else (
                "the incumbent was not promoted since the last block (promoted since: " + ", ".join(promoted) + "; --all-promoted plays them)")
            print(f"[{utc_now()}] promotion block: {why}", flush=True)
            (out / "result.json").write_text(json.dumps({"idle": why, "agents": {}, "trigger": "promotion", "started_epoch": t0,
                                                         "promoted_not_played": left_out}), encoding="utf-8")
            return 0
        a.runs = night_runs(listed, a.runs, a.runs_promoted)  # a generation promoted since the last block: the full block, for the incumbent alone
        print(f"[{utc_now()}] {a.trigger_used} block {a.date}: {a.runs} runs per agent, plays {', '.join(ag['label'] for ag in todo)} "
              f"({'promoted since the last block: ' + ', '.join(promoted) if promoted else 'no promotion since the last block'}"
              + (f"; not played, --all-promoted would: {', '.join(left_out)}" if left_out else "") + ")", flush=True)
        for ag in todo:
            try:
                agents[ag["label"]] = play_agent(a, ag, a.date, date_dir)
            except (OSError, ValueError, KeyError) as e:  # one agent's problem (a shared seed, a missing directory) does not stop the others
                agents[ag["label"]] = {"date": a.date, "agent": ag["label"], "error": f"{type(e).__name__}: {e}"[:300]}
                print(f"[{utc_now()}] {ag['label']}: {agents[ag['label']]['error']}", flush=True)
        good = [v for v in agents.values() if "error" not in v]
        inc = good[0] if good else {}
        res = {"date": a.date, "trigger": a.trigger_used, "started_epoch": t0, "engine": a.engine, "runs_per_agent": a.runs, "emu_runs_per_agent": emu_runs_tonight(a.emu_runs, a.runs) or a.runs,
               "all_promoted": bool(getattr(a, "all_promoted", False)), "promoted_not_played": left_out,
               "agents": agents, "wall_s": round(time.time() - t0, 1),
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
