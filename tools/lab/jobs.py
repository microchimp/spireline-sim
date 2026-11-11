"""The agent-era job kinds of the CPU schedule (docs/PLAN.md section 7): selfplay, train, score-real, nightly-real, bench-env, bench-emu.

Each kind is data: a default command, a limit, whether it is on, and the input that makes it ready. ``LAB/jobs.json``
(optional, read at every choice, keys starting with "_" are notes) overrides any of it per kind:

  {"selfplay": {"enabled": true, "cmd": ["{python}", "rl/run_train/cli.py", "..."], "share": 0.5, "timeout_min": 60}}

The shipped defaults change nothing for a lab that has no jobs.json: selfplay and the benches are off, and train and
score-real are on but only ever ready when their marker file exists (LAB/train/ready-<gen>, ready-<gen>-r<n> for a retrain after a rejection or ready-<gen>-g<target> for a candidate that is generation <target>, LAB/score/pending).

Priority ladder (lower number first; what the runner's choose() does with it is in runner.py):

  1 presort   1.5 nightly-real   2 score-real   3 train   4 aimed fights   5 runs / rank   6 selfplay   7 the rest, bench-* nightly
  8 teacher (the run-level teacher, the filler's slot: the last thing the machine does; section "the teacher job" below)

The real-game checks come first (owner, 2026-11-06: the real-game check is the bottleneck): when LAB/nightly/pending is written (by
hand, by the orchestrator, after a promotion; no code writes it) the next pick after the running job is the nightly block, then the
fixed-seed score (LAB/score/pending, which a promotion writes).

nightly-real is a kind of its own (D8, D33 addendum 2): score-real is one command on one marker, the nightly block is a driver
(rl/nightly_real.py) that plays several agents under the QUIET.lock, so the due rule and the window markers sit here and the
work sits in the driver. Ready (2026-11-06): LAB/nightly/pending exists (hand-written; the block plays the generations promoted
since the last block, `trigger` promotion) or LAB/nightly/last-ok is older than `every_h` (the clock stays as the weekly copy-accuracy
watch, `trigger` clock). Skipped, with the reason, while LAB/HALT or LAB/PAUSE is up, LAB/nightly/active is fresh (the last block has not
finished) or LAB/score/active is fresh; nothing is pre-empted. The pending marker is consumed when the block ends (so an interrupted or
superseded block is still wanted), unless it was written after the block took its list (the marker is newer than the block's start);
last-ok is stamped with the block's START time, so a promotion that landed while the block played is "since the last block" for the
next one.

A job's command may write ``<job>/result.json`` (an object); its keys named in mix.YIELD_COLUMNS become the job's
yield.tsv columns, and the bench kinds also store it under readiness.json ``bench.env`` / ``bench.emu``.
"""

from __future__ import annotations

import json
import os
import re
import time
from pathlib import Path
from typing import Any

NEW_KINDS = ("selfplay", "train", "score-real", "nightly-real", "bench-env", "bench-emu", "teacher")
BENCHES = ("bench-env", "bench-emu")
# The kinds that exist only because self-play does: self-play, the training (and the promotion it runs) of what it recorded, and the
# real-game score of what training promoted. `selfplay.share` is judged on their CPU together (mix.loop_cpu), not on self-play alone.
LOOP_KINDS = ("selfplay", "train", "score-real")  # nightly-real is outside the loop share: it scores the incumbent whether or not self-play ran (D29 accounting)
PRIORITY = {"presort": 1, "nightly-real": 1.5, "score-real": 2, "train": 3, "fights-aimed": 4, "runs": 5, "selfplay": 6, "rest": 7,
            "bench-env": 7, "bench-emu": 7, "teacher": 8}
SCORE_WINDOW_H = 4.0  # the longest a score-real window may run
ACTIVE_STALE_H = SCORE_WINDOW_H + 1.0  # an `active` marker older than this is a crash's leftover
NIGHTLY_WINDOW_H = 15.0  # the longest a nightly block may run (all agents): the job's 900-minute cap (2026-11-06; a 2,000-run block of the x16 player is about 7.6 h)
NIGHTLY_ACTIVE_STALE_H = NIGHTLY_WINDOW_H + 1.0
NIGHTLY_RETRY_H = 1.0  # after a block that was skipped (the QUIET.lock never freed), the next try is this much later


def _placeholder(kind: str, entry: str) -> list[str]:
    """A command that only prints what it stands for; A2 sets the real one in LAB/jobs.json."""
    return ["{python}", "-c", f"print('placeholder: set LAB/jobs.json {kind}.cmd; entry point {entry}')"]


DEFAULTS: dict[str, dict[str, Any]] = {
    "selfplay": {"enabled": False, "cmd": _placeholder("selfplay", "rl/run_train/cli.py"), "cwd": "{tree}", "timeout_min": 60,
                 "share": 0.0, "share_days": 7.0, "needs": []},
    "train": {"enabled": True, "cmd": _placeholder("train", "rl/run_train/cli.py"), "cwd": "{tree}", "timeout_min": 60, "gpu": True,
              "needs": []},
    "score-real": {"enabled": True, "cmd": _placeholder("score-real", "rl/score_real.py"), "cwd": "{tree}", "timeout_min": 240,
                   "procs": 14, "weekly": False, "weekly_days": 7.0, "needs": []},
    "nightly-real": {"enabled": False, "cmd": _placeholder("nightly-real", "rl/nightly_real.py"), "cwd": "{tree}", "timeout_min": 900,
                     "procs": 12, "every_h": 20.0, "needs": []},
    "bench-env": {"enabled": False, "cmd": _placeholder("bench-env", "the envproto bench on branch env/leak-probe"), "cwd": "{tree}",
                  "timeout_min": 40, "procs": 12, "every_h": 20.0, "needs": []},
    "bench-emu": {"enabled": False, "cmd": ["{python}", "rl/bench_emu.py"], "cwd": "{tree}", "timeout_min": 30, "every_h": 20.0,
                  "needs": ["{tree}/rl/bench_emu.py"]},
    # The run-level teacher (D56, D57 part 3; docs/lab/TEACHER_JOB.md). Off until LAB/jobs.json switches it on.
    "teacher": {"enabled": False, "cwd": "{tree}", "timeout_min": 240, "procs": 4, "cores": 4, "core_hours_per_day": 24.0, "block_min": 60,
                "presort_horizon_min": 60.0, "quiet_ignore": ["gym-loop", "lab-score"], "store": "D:/spireline/referee/run-teacher",
                "cmd": ["{python}", "rl/run_teacher.py", "job", "--lab", "{lab}", "--store", "D:/spireline/referee/run-teacher",
                        "--agent-root", "D:/spireline/referee/agent", "--judge", "D:/spireline/referee/agent/judge/v0/judge.pt", "--lib", "{tree}/out",
                        "--source", "new", "--procs", "4", "--block-min", "60", "--budget-h", "24", "--result", "{job}/result.json"],
                "needs": ["{tree}/rl/run_teacher.py", "{tree}/out/Sts2Emulator.dll"]},
}
BENCH_KEY = {"bench-env": "env", "bench-emu": "emu"}

# The fidelity-era job kinds and report steps are off unless LAB/jobs.json switches them on (D57 addendum 1, 2026-11-09):
#   "rotation": ["fights", "probes", ...]      which of FIDELITY_KINDS the runner may pick; absent or empty: none of them
#   "reports": {"checks": {"enabled": true, "every_h": 168}, "groups": true, ...}   which of REPORT_STEPS run; absent: none
FIDELITY_KINDS = ("fights", "probes", "mutation", "invariants", "runs")
REPORT_STEPS = ("checks", "corpus-import", "groups", "kits")


def rotation_kinds(doc: Any) -> tuple[tuple[str, ...], list[str]]:
    """(the fidelity-era kinds `doc["rotation"]` lets the runner pick, in FIDELITY_KINDS order; the names in it that are not kinds). The entry
    is a list of names (a dict with a `kinds` list is read the same); anything else, an empty list and a missing entry are none."""
    raw = doc.get("rotation") if isinstance(doc, dict) else None
    if isinstance(raw, dict):
        raw = raw.get("kinds")
    names = [str(x) for x in raw] if isinstance(raw, list) else []
    return tuple(k for k in FIDELITY_KINDS if k in names), [n for n in names if n not in FIDELITY_KINDS]


def reports_config(doc: Any) -> dict[str, dict[str, Any]]:
    """{step: {"enabled": bool, "every_h": hours or None}} for every REPORT_STEPS name from `doc["reports"]` (a step is off when it is not
    named; its entry is a bool or {"enabled": bool, "every_h": N}; `every_h` is the least time between two runs, used by `checks`)."""
    raw = doc.get("reports") if isinstance(doc, dict) else None
    raw = raw if isinstance(raw, dict) else {}
    out: dict[str, dict[str, Any]] = {}
    for step in REPORT_STEPS:
        v = raw.get(step)
        on = bool(v.get("enabled")) if isinstance(v, dict) else v is True
        every = v.get("every_h") if isinstance(v, dict) else None
        every = float(every) if isinstance(every, (int, float)) and not isinstance(every, bool) and every > 0 else None
        out[step] = {"enabled": on, "every_h": every}
    return out


def read_lenient(p: Path) -> Any:
    try:
        return json.loads(p.read_bytes().decode("utf-8", errors="replace").lstrip("﻿").replace("\x00", ""))
    except (OSError, ValueError):
        return None


def load_config(lab: Path) -> dict[str, dict[str, Any]]:
    """The settings of every new kind: the defaults, with LAB/jobs.json's per-kind keys laid over them."""
    return config_from(read_lenient(Path(lab) / "jobs.json"))


def config_from(doc: Any) -> dict[str, dict[str, Any]]:
    """load_config for a jobs.json already read (only a dict changes the defaults)."""
    cfg = {k: dict(v) for k, v in DEFAULTS.items()}
    if isinstance(doc, dict):
        for k, v in doc.items():
            if k in cfg and isinstance(v, dict):
                cfg[k].update({a: b for a, b in v.items() if not str(a).startswith("_")})
    return cfg


def _age_h(p: Path) -> float | None:
    try:
        return (time.time() - p.stat().st_mtime) / 3600
    except OSError:
        return None


# a recipe tag (rl/gen_loop.TAG): `ready-11-g12-imit` is the candidate gen12/candidate-imit
TAG = r"[a-z]{3}[a-z0-9]*(?:-[a-z0-9]+)*"


def gen_key(name: str) -> tuple[int, int, int, int, str]:
    """Marker order: by generation, then the plain and retrain forms by index, then the target forms by target and attempt
    (``ready-4`` < ``ready-4-r1`` < ``ready-4-g6`` < ``ready-4-g6-a2`` < ``ready-4-g6-imit`` < ``ready-12``)."""
    m = re.search(r"-(\d+)(?:-r(\d+)|-g(\d+)(?:-a(\d+)|-" + TAG + r")?)?$", name)
    if not m:
        return (1 << 30, 0, 0, 0, name)
    return (int(m.group(1)), 1 if m.group(3) else 0, int(m.group(3) or m.group(2) or 0), int(m.group(4) or 0), name)


def train_markers(lab: Path) -> list[Path]:
    d = Path(lab) / "train"
    return sorted(d.glob("ready-*"), key=lambda p: gen_key(p.name)) if d.is_dir() else []


def score_active(lab: Path) -> bool:
    """A score-real window is open (its marker is fresh): fights jobs must not start."""
    a = _age_h(Path(lab) / "score" / "active")
    return a is not None and a < ACTIVE_STALE_H


def nightly_active(lab: Path) -> bool:
    """A nightly-real window is open (its marker is fresh): nothing of the lab may run beside the real-game arms. Fresh is the nightly
    window's own span (NIGHTLY_ACTIVE_STALE_H), not the score window's: a block of the x16 player runs 7 to 15 hours."""
    a = _age_h(Path(lab) / "nightly" / "active")
    return a is not None and a < NIGHTLY_ACTIVE_STALE_H


GPU_OK_FILE = "D:/spireline/referee/GPU_OK"  # rl/gen0.py GPU_OK_FILE: the lines of the lanes that allow a GPU fit beside the real-game arm
CONSENT_TAG = "lab-nightly"  # rl/nightly_real.py TAG: the line a nightly-real job keeps first in GPU_OK while its real-game arm plays


def consent_path(entry: dict | None) -> str:
    """The GPU_OK file a nightly-real job's driver puts its consent line in, from the entry's settings: the `--gpu-ok-file` of its `cmd`
    (as `--gpu-ok-file P` or `--gpu-ok-file=P`; a value that still holds a {token} is not a path yet), else its `gpu_ok_file` key, else
    the default the driver uses (GPU_OK_FILE)."""
    entry = entry or {}
    cmd = entry.get("cmd")
    toks = [str(x) for x in cmd] if isinstance(cmd, (list, tuple)) else str(cmd).split() if isinstance(cmd, str) else []
    for i, t in enumerate(toks):
        v = t.split("=", 1)[1] if t.startswith("--gpu-ok-file=") else toks[i + 1] if t == "--gpu-ok-file" and i + 1 < len(toks) else None
        if v and "{" not in v:
            return v
    return str(entry.get("gpu_ok_file") or GPU_OK_FILE)


def clear_consent(path: str | Path, tag: str = CONSENT_TAG) -> bool:
    """Remove the lines equal to `tag` from the GPU_OK file `path`, the other lines kept and the file deleted when nothing else is left
    in it (the rule of gen0.gpu_ok_remove, which the runner does not import). True when a line was removed; best effort."""
    p = Path(path)
    try:
        if not p.is_file():
            return False
        have = [h for h in p.read_text(encoding="utf-8").splitlines() if h.strip()]
        rest = [h for h in have if h.strip() != tag]
        if len(rest) == len(have):
            return False
        if not rest:
            p.unlink()
            return True
        tmp = p.with_name(p.name + ".tmp")
        tmp.write_text("".join(x + "\n" for x in rest), encoding="utf-8")
        os.replace(tmp, p)
        return True
    except OSError:
        return False


def due(lab: Path, kind: str, every_h: float) -> bool:
    a = _age_h(Path(lab) / kind / "last-ok")
    return a is None or a >= every_h


def nightly_pending_time(lab: Path) -> float | None:
    """Epoch seconds LAB/nightly/pending was last written (by hand: no code writes it), None when there is no such marker."""
    try:
        return (Path(lab) / "nightly" / "pending").stat().st_mtime
    except OSError:
        return None


def nightly_wanted(lab: Path, every_h: float) -> bool:
    """A nightly block is wanted: LAB/nightly/pending was written or the clock (last-ok older than `every_h`) says so."""
    return nightly_pending_time(lab) is not None or due(lab, "nightly", every_h)


def consume_nightly_pending(lab: Path, started: float | None) -> bool:
    """Remove LAB/nightly/pending after a block that took its list of promotions at `started` (epoch seconds), unless the marker was
    written after that: the next block plays it. `started` None (a result that does not say) removes it. True when removed."""
    t = nightly_pending_time(lab)
    if t is None or (started is not None and t > started):
        return False
    (Path(lab) / "nightly" / "pending").unlink(missing_ok=True)
    return True


def nightly_blocked(lab: Path) -> str | None:
    """Why a due nightly block must wait (None: it may start). The runner's guards (PAUSE, HALT, load, disk) stop every job
    alike; main.lock holds every new pick but stops only a running presort (a running nightly carries on under it). They are named
    here too so the skip carries its reason in runner.log, and so a test can state the rule."""
    lab = Path(lab)
    if (lab / "HALT").exists():
        return "LAB/HALT is up"
    if (lab / "PAUSE").exists():
        return "LAB/PAUSE is up"
    a = _age_h(lab / "nightly" / "active")
    if a is not None and a < NIGHTLY_ACTIVE_STALE_H:
        return "the previous nightly block has not finished (LAB/nightly/active is fresh)"
    if score_active(lab):
        return "a score-real window is open (LAB/score/active)"
    s = _age_h(lab / "nightly" / "last-skip")
    if s is not None and s < NIGHTLY_RETRY_H:
        return f"the last nightly attempt was skipped (the QUIET.lock stayed taken); next try {NIGHTLY_RETRY_H:g} h after it"
    return None


def unready(kind: str, lab: Path, cfg: dict[str, Any]) -> str | None:
    """None when the kind's input is ready, else why not. The guards are separate (they apply to every kind alike)."""
    if not cfg.get("enabled"):
        return f"{kind} is off (LAB/jobs.json {kind}.enabled)"
    if kind == "selfplay":
        return None
    if kind == "train":
        return None if train_markers(lab) else "no LAB/train/ready-<gen> marker"
    if kind == "score-real":
        if (Path(lab) / "score" / "pending").exists():
            return None
        if cfg.get("weekly") and due(lab, "score", 24 * float(cfg.get("weekly_days", 7.0))):
            return None
        return "no LAB/score/pending marker and the weekly block is not due"
    if kind == "nightly-real":
        if not nightly_wanted(lab, float(cfg.get("every_h", 20.0))):
            return "ran within the last night"
        return nightly_blocked(lab)
    if kind in BENCHES:
        return None if due(lab, kind, float(cfg.get("every_h", 20.0))) else "ran within the last night"
    if kind == TEACHER:
        return teacher_hold(lab, cfg, starting=True)
    return f"unknown kind {kind}"


def params_for(kind: str, lab: Path, cfg: dict[str, Any]) -> dict[str, Any]:
    p: dict[str, Any] = {"strategy": kind, "cmd": list(cfg["cmd"]), "cwd": cfg.get("cwd", "{tree}"),
                         "timeout_min": float(cfg.get("timeout_min", 60)), "needs": list(cfg.get("needs") or [])}
    for k in ("procs", "gpu"):
        if k in cfg:
            p[k] = cfg[k]
    if kind == "train":
        ms = train_markers(lab)
        p["marker"] = ms[0].name if ms else ""
        p["generation"] = p["marker"].removeprefix("ready-")
        target = re.fullmatch(r"ready-(\d+)-g(\d+)(?:-a(\d+)|-(" + TAG + r"))?", p["marker"])
        if target:  # `ready-<gen>-g<target>[-a<n>]`: the shards of <gen> train a candidate that is generation <target> (D33 addendum (b))
            p["generation"], p["candidate_generation"] = target.group(1), int(target.group(2))
            if target.group(3):  # the n-th attempt at that number: a built candidate in gen<target>/candidate-a<n>
                p["attempt"] = int(target.group(3))
            if target.group(4):  # a tagged recipe: a built candidate in gen<target>/candidate-<tag>
                p["tag"] = target.group(4)
    return p


# ---------------------------------------------------------------- what a finished command leaves behind

def read_result(job_dir: Path) -> dict[str, Any]:
    doc = read_lenient(Path(job_dir) / "result.json")
    return doc if isinstance(doc, dict) else {}


def _atomic(p: Path, text: str) -> None:
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8")
    os.replace(tmp, p)


def store_bench(readiness: Path, kind: str, result: dict[str, Any]) -> None:
    """readiness.json `bench.<env|emu>` = result; every other key (and the other bench) stays."""
    doc = read_lenient(readiness)
    doc = doc if isinstance(doc, dict) else {}
    bench = doc.get("bench") if isinstance(doc.get("bench"), dict) else {}
    bench[BENCH_KEY[kind]] = result
    doc["bench"] = bench
    _atomic(readiness, json.dumps(doc, indent=1))


def store_nightly(readiness: Path, result: dict[str, Any]) -> None:
    """readiness.json `nightly` = the last block's numbers per agent, with the date; every other key stays."""
    doc = read_lenient(readiness)
    doc = doc if isinstance(doc, dict) else {}
    doc["nightly"] = {k: result[k] for k in ("date", "engine", "runs_per_agent", "agents") if k in result}
    _atomic(readiness, json.dumps(doc, indent=1))


def consume_marker(marker: Path | str) -> Path | None:
    """`ready-<x>` becomes `done-<x>` (what a finished round leaves: `ready-9`, `ready-9-r1`, `ready-9-g11`, `ready-11-g12-a2`); the new path, or None
    when there is no such file. Used by `finish` and by `gen0.py train` when it stops a round the review guard could not judge."""
    m = Path(marker)
    if not m.exists():
        return None
    done = m.with_name("done-" + m.name.removeprefix("ready-"))
    os.replace(m, done)
    return done


def finish(kind: str, lab: Path, job_dir: Path, params: dict[str, Any]) -> str | None:
    """After the command succeeded: consume the input marker, stamp the night, store the bench. A problem string or None."""
    lab = Path(lab)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n"
    try:
        if kind == "train" and params.get("marker"):
            consume_marker(lab / "train" / params["marker"])
        elif kind == "score-real":
            (lab / "score" / "pending").unlink(missing_ok=True)
            _atomic(lab / "score" / "last-ok", stamp)
        elif kind == "nightly-real":
            res = read_result(job_dir)
            started = res.get("started_epoch")
            started = float(started) if isinstance(started, (int, float)) and not isinstance(started, bool) and 0 < started <= time.time() else None
            if res.get("skipped"):  # the QUIET.lock never freed: not a night; try again later (a pending marker stays)
                _atomic(lab / "nightly" / "last-skip", stamp)
            elif res.get("idle"):  # a promotion block with no generation to play (the marker outlived its promotion): nothing was played, nothing stamped
                consume_nightly_pending(lab, started)
            else:
                if res.get("agents"):
                    store_nightly(lab.parent / "readiness.json", res)
                _atomic(lab / "nightly" / "last-ok", stamp)
                if started is not None:  # the block's start, not its end: a promotion that landed while it played is "since the last block" for the next
                    os.utime(lab / "nightly" / "last-ok", (started, started))
                (lab / "nightly" / "last-skip").unlink(missing_ok=True)
                consume_nightly_pending(lab, started)
        elif kind in BENCHES:
            res = read_result(job_dir)
            if res:
                store_bench(lab.parent / "readiness.json", kind, res)
            _atomic(lab / kind / "last-ok", stamp)
    except OSError as e:
        return f"{kind}: could not update its markers ({type(e).__name__})"
    return None


# ---------------------------------------------------------------- score-real: the ranking step

def records_dir(job_dir: Path) -> Path | None:
    """Where the score-real command left its lockstep records: result.json `out` (a path, absolute or in the job folder), else the
    job folder or the first folder under it that holds divergences.jsonl."""
    job_dir = Path(job_dir)
    out = read_result(job_dir).get("out")
    cands = ([job_dir / str(out)] if out else []) + [job_dir]
    cands += sorted(p.parent for p in job_dir.rglob("divergences.jsonl"))
    return next((c for c in cands if (c / "divergences.jsonl").is_file()), None)


def rank_score_real(job_dir: Path) -> str | None:
    """The analysis step of score-real: ranked.json and top20.md into the job folder, and divergences, groups, arch_share into
    result.json (the yield columns). A problem string or None; no records is not a problem (the command may have run nothing)."""
    from tools.lab import rank  # noqa: PLC0415
    job_dir = Path(job_dir)
    src = records_dir(job_dir)
    if src is None:
        return None
    try:
        s = rank.write(src, job_dir, emu_dir=rank.paired_emu_dir(src) or rank.paired_emu_dir(job_dir))["summary"]
        res = read_result(job_dir)
        res.update(divergences=s["state_divergences"], groups=s["groups"], arch_share=s["arch_share"])
        _atomic(job_dir / "result.json", json.dumps(res, indent=1))
    except (OSError, ValueError, KeyError) as e:
        return f"score-real: ranking failed ({type(e).__name__})"
    return None


# ---------------------------------------------------------------- the teacher job (D56, D57 part 3; rl/run_teacher.py `job`)

TEACHER = "teacher"
TEACHER_FORK = "D:/spireline/fork/sts2"  # the checkout whose local main the presort rule looks at (the runner's own --fork)
TEACHER_RESERVE_H = 2.0  # a run that started and wrote no end line (its job was killed) still counts its estimate for this long
TEACHER_EST_CPU_S = 1100.0  # a teacher run's CPU seconds when the ledger has no finished run to average (2026-11-11 measurement: 1,090 to 1,700)
TEACHER_MAIN_CACHE_S = 20.0
_main_cache: dict[str, tuple[float, str | None]] = {}


def teacher_ledger(store: Path | str) -> Path:
    """The job's CPU ledger: one JSON line per event (a run starting, a run ending with its CPU seconds, a job ending), all generations."""
    return Path(store) / "ledger.jsonl"


def read_ledger(path: Path | str) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    try:
        text = Path(path).read_text(encoding="utf-8", errors="replace")
    except OSError:
        return rows
    for line in text.splitlines():
        try:
            r = json.loads(line)  # a line cut by a kill does not parse and is dropped
        except ValueError:
            continue
        if isinstance(r, dict):
            rows.append(r)
    return rows


def append_ledger(path: Path | str, row: dict[str, Any]) -> None:
    """One event line, written in one call (the job's worker processes append to the same file)."""
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    row = {"t": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), **row}
    with open(p, "a", encoding="utf-8", newline="\n") as fh:
        fh.write(json.dumps(row) + "\n")


def _stamp(s: Any) -> float | None:
    try:
        import calendar  # noqa: PLC0415

        return float(calendar.timegm(time.strptime(str(s), "%Y-%m-%dT%H:%M:%SZ")))
    except (ValueError, TypeError):
        return None


def teacher_spent_s(rows: list[dict[str, Any]], day: str, now: float, est_s: float | None = None, reserve_h: float = TEACHER_RESERVE_H) -> float:
    """CPU seconds the teacher job has spent on the UTC day `day` ("YYYY-MM-DD"): the CPU of the runs whose end line is dated that day,
    plus an estimate (`est_s`, else the mean of the finished runs on the ledger, else TEACHER_EST_CPU_S) for each run that has started and
    has no end line, unless its job wrote its own end line or it started more than `reserve_h` ago. A run in flight is therefore paid
    for before it ends, so a day's budget is not overspent by the runs that were running when it ran out."""
    ends = [r for r in rows if r.get("ev") == "end" and isinstance(r.get("cpu"), (int, float))]
    spent = sum(float(r["cpu"]) for r in ends if str(r.get("t", ""))[:10] == day)
    if est_s is None:
        est_s = sum(float(r["cpu"]) for r in ends) / len(ends) if ends else TEACHER_EST_CPU_S
    ended = {(r.get("job"), r.get("seed")) for r in rows if r.get("ev") == "end"}
    job_over = {r.get("job") for r in rows if r.get("ev") == "job_end"}
    open_runs: dict[tuple, float] = {}
    for r in rows:
        if r.get("ev") != "start":
            continue
        key = (r.get("job"), r.get("seed"))
        t = _stamp(r.get("t"))
        if key in ended or r.get("job") in job_over or t is None or now - t > reserve_h * 3600 or str(r.get("t", ""))[:10] != day:
            continue
        open_runs[key] = float(r.get("est") or est_s)
    return spent + sum(open_runs.values())


def teacher_budget_left_h(cfg: dict[str, Any], rows: list[dict[str, Any]], now: float | None = None) -> float:
    """Core-hours the day's budget (`core_hours_per_day`, default 24) has left, never below 0."""
    now = time.time() if now is None else now
    day = time.strftime("%Y-%m-%d", time.gmtime(now))
    return max(0.0, float(cfg.get("core_hours_per_day", 24.0)) - teacher_spent_s(rows, day, now) / 3600.0)


def _git_main(fork: str, branch: str = "main") -> str | None:
    import subprocess  # noqa: PLC0415

    now = time.time()
    got = _main_cache.get(fork)
    if got is not None and now - got[0] < TEACHER_MAIN_CACHE_S:
        return got[1]
    sha = None
    try:
        r = subprocess.run(["git", "-C", fork, "rev-parse", "--verify", f"refs/heads/{branch}"], capture_output=True, text=True, timeout=20, check=False)
        if r.returncode == 0:
            sha = r.stdout.strip() or None
    except (OSError, subprocess.SubprocessError):
        sha = None
    _main_cache[fork] = (now, sha)
    return sha


def presort_due_soon(state: dict[str, Any] | None, main_sha: str | None, now: float, horizon_s: float, gap_min: float = 60.0,
                     last_end: float | None = None) -> str | None:
    """Why a block must not start: the runner would choose a presort within `horizon_s` (the filler's rule, D55 addendum 18). That is when
    the fork's local main (`main_sha`) has moved past the presorted tree and the last presort ended `gap_min` minutes ago or more, or will
    have by then (a build that failed waits out its retry first). `state` is the runner's LAB/state.json; `last_end` the end of the last
    presort when the state does not say."""
    st = state or {}
    if not main_sha or (st.get("last_presorted_sha") == main_sha and st.get("tree_sha") == main_sha):
        return None
    ended = st.get("last_presort_end") or last_end or 0.0
    due = float(ended) + max(0.0, gap_min) * 60
    bad = (st.get("bad_shas") or {}).get(main_sha)
    if isinstance(bad, dict):
        due = max(due, float(bad.get("retry_after", 0)))
    if due - now > horizon_s:
        return None
    return f"a presort of {main_sha[:7]} is due within {horizon_s / 60:g} min (main moved past the presorted tree)"


def _last_presort_end(lab: Path) -> float | None:
    ends = []
    try:
        for d in (Path(lab) / "runs").glob("*-presort"):
            try:
                ends.append((d / "done.rc").stat().st_mtime)
            except OSError:
                continue
    except OSError:
        return None
    return max(ends, default=None)


def teacher_hold(lab: Path | str, cfg: dict[str, Any] | None = None, *, starting: bool = True, now: float | None = None,
                 quiet_path: Path | str | None = None, state: dict[str, Any] | None = None, main_sha: str | None = "",
                 ledger_rows: list[dict[str, Any]] | None = None) -> str | None:
    """Why the teacher job may not start (`starting`) or must stop now (a running block); None when it may run. The filler's rules, kept
    here as one function of the lab folder so the runner's pick (`unready`) and the job's own poll (`rl/run_teacher.py job`) agree:

      HALT or PAUSE up; a night running or due (`cores.night_hold`: none within `night_margin_min` of the night's due time, a running
      block stops two minutes before it); a live QUIET.lock line that is not in `quiet_ignore` (default: the loop's and the daytime
      score's; `lab-train`, `w3-blocks` and every other tag make the job yield); judgement first: a train marker or a score-real
      pending marker that is waiting for the machine, a presort due within `presort_horizon_min`, and the day's core-hour budget spent (these three
      only when starting: a block running when a marker is written or the budget runs out ends within its `block_min`, its workers start no
      run the budget has no room for).

    `main_sha` "" means look (the fork's local main, cached 20 s); None means unknown (no presort rule)."""
    from tools.lab import cores  # noqa: PLC0415 -- cores imports this module

    lab = Path(lab)
    cfg = cfg if cfg is not None else config_from(read_lenient(lab / "jobs.json"))[TEACHER]
    now = time.time() if now is None else now
    doc = cores.load_doc(lab)
    if (lab / "HALT").exists():
        return "LAB/HALT is up"
    if (lab / "PAUSE").exists():
        return "LAB/PAUSE is up"
    why = cores.night_hold(lab, doc, now, starting=starting)
    if why:
        return why
    qp = Path(quiet_path) if quiet_path else lab.parent / "QUIET.lock"
    ignore = cfg.get("quiet_ignore")
    ignore = list(ignore) if isinstance(ignore, (list, tuple)) else list(cores.DEFAULT_QUIET_IGNORE)
    tag = cores.quiet_blocker(qp, {"selfplay": {"quiet_ignore": ignore}}, now)
    if tag:
        return f"QUIET.lock is held by {tag}"
    if starting:
        full = config_from(doc)
        if full["train"].get("enabled") and train_markers(lab):
            return f"judgement first: {train_markers(lab)[0].name} is waiting"
        if full["score-real"].get("enabled") and (lab / "score" / "pending").exists():
            return "judgement first: LAB/score/pending is waiting"
        if state is None:
            state = read_lenient(lab / "state.json")
        sha = _git_main(str(cfg.get("fork") or TEACHER_FORK)) if main_sha == "" else main_sha
        st = state if isinstance(state, dict) else {}
        why = presort_due_soon(st, sha, now, float(cfg.get("presort_horizon_min", 60.0)) * 60, float(cfg.get("presort_gap_min", 60.0)),
                               None if st.get("last_presort_end") else _last_presort_end(lab))
        if why:
            return why
        rows = ledger_rows if ledger_rows is not None else read_ledger(teacher_ledger(cfg.get("store") or "D:/spireline/referee/run-teacher"))
        if teacher_budget_left_h(cfg, rows, now) <= 0.0:  # a running block is never cut for it: its workers start no run the budget has no room for
            return (f"the day's budget is spent ({float(cfg.get('core_hours_per_day', 24.0)):g} core-hours, "
                    f"UTC day {time.strftime('%Y-%m-%d', time.gmtime(now))})")
    return None
