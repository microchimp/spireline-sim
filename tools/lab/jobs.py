"""The agent-era job kinds of the CPU schedule (docs/PLAN.md section 7): selfplay, train, score-real, nightly-real, bench-env, bench-emu, teacher, student.

Each kind is data: a default command, a limit, whether it is on, and the input that makes it ready. ``LAB/jobs.json``
(optional, read at every choice, keys starting with "_" are notes) overrides any of it per kind:

  {"selfplay": {"enabled": true, "cmd": ["{python}", "rl/run_train/cli.py", "..."], "share": 0.5, "timeout_min": 60}}

The shipped defaults change nothing for a lab that has no jobs.json: selfplay and the benches are off, and train and
score-real are on but only ever ready when their marker file exists (LAB/train/ready-<gen>, ready-<gen>-r<n> for a retrain after a rejection or ready-<gen>-g<target> for a candidate that is generation <target>, LAB/score/pending).

Priority ladder (lower number first; what the runner's choose() does with it is in runner.py):

  1 presort   1.5 nightly-real   2 score-real   3 train   4 aimed fights   5 runs / rank   6 selfplay   7 the rest, bench-* nightly
  8 teacher (the run-level teacher, the filler's slot: the last thing the machine does; section "the teacher job" below)
  3.5 student (the run student's fit, legs and candidate, right after train; section "the student job" below, docs/lab/RUN_LOOP.md)

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

NEW_KINDS = ("selfplay", "train", "score-real", "nightly-real", "bench-env", "bench-emu", "teacher", "student")
BENCHES = ("bench-env", "bench-emu")
# The kinds that exist only because self-play does: self-play, the training (and the promotion it runs) of what it recorded, and the
# real-game score of what training promoted. `selfplay.share` is judged on their CPU together (mix.loop_cpu), not on self-play alone.
LOOP_KINDS = ("selfplay", "train", "score-real")  # nightly-real is outside the loop share: it scores the incumbent whether or not self-play ran (D29 accounting)
PRIORITY = {"presort": 1, "nightly-real": 1.5, "score-real": 2, "train": 3, "fights-aimed": 4, "runs": 5, "selfplay": 6, "rest": 7,
            "bench-env": 7, "bench-emu": 7, "teacher": 8, "student": 3.5}
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
                        "--agent-root", "D:/spireline/referee/agent", "--judge", "D:/spireline/referee/agent/judge/v2b/judge-board.pt", "--lib", "{tree}/out",
                        "--source", "new", "--procs", "4", "--block-min", "60", "--budget-h", "24", "--result", "{job}/result.json"],
                "needs": ["{tree}/rl/run_teacher.py", "{tree}/out/Sts2Emulator.dll"]},
    # The run student (D56, D57 part 3, D58 C4; docs/lab/RUN_LOOP.md). Off until LAB/jobs.json switches it on. `{student_*}` tokens are
    # filled when the job is made (`params_for`): the incumbent's prior (or student), the human prior the teacher decided with (the flag and its token are
    # left out when the fit starts from that prior itself), the candidate's number and tag.
    "student": {"enabled": False, "cwd": "{tree}", "timeout_min": 180, "procs": 4, "cores": 4, "gpu": True, "new_screens": 10000,
                "weekly_days": 7.0, "weekly_min_screens": 1000, "quiet_ignore": ["gym-loop", "lab-score"], "ok_rcs": [0, 3],
                "shards_root": "D:/spireline/referee/run-teacher", "human_shards": "D:/spireline/referee/human-prior/decisions-v1",
                "agent_root": "D:/spireline/referee/agent", "check_timeout_s": 900,
                "cmd": ["D:/spireline/venv-gpu/Scripts/python.exe", "tools/lab/student_job.py", "--lab", "{lab}",
                        "--shards-root", "D:/spireline/referee/run-teacher", "--prior", "{student_prior}", "--teacher-prior", "{student_teacher_prior}",
                        "--human-shards", "D:/spireline/referee/human-prior/decisions-v1", "--out", "{job}/student",
                        "--base", "D:/spireline/referee/agent/current.json", "--agent-root", "D:/spireline/referee/agent",
                        "--target", "{student_target}", "--tag", "{student_tag}", "--records", "D:/spireline/referee/run-teacher",
                        "--result", "{job}/result.json"],
                "needs": ["{tree}/tools/lab/student_job.py", "{tree}/rl/run_student.py", "D:/spireline/venv-gpu/Scripts/python.exe"]},
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
    if kind == STUDENT:
        return student_hold(lab, cfg)
    return f"unknown kind {kind}"


def held_reason(kind: str, lab: Path | str, cfg: dict[str, Any], why: str | None, now: float | None = None) -> str | None:
    """`why` (what `unready` said) when the student or the teacher is wanted and held, so the runner says it once; None when the kind is off, ready, or
    only not due yet. The teacher is wanted whenever it is on (the filler). The student is wanted once its trigger is met by the screens recorded
    (the count `check` makes is the next thing it asks, and a `check` that fails is what the reason says); a student short of the trigger, or one whose
    `check` keeps too few labelled screens, is passed over quietly."""
    if not why or not cfg.get("enabled") or kind not in (STUDENT, TEACHER):
        return None
    if kind == STUDENT:
        if LABELLED_SHORT in why:
            return None
        now = time.time() if now is None else now
        st = read_student_state(lab)
        recorded, _ = student_recorded_screens(Path(str(cfg.get("shards_root") or DEFAULTS[STUDENT]["shards_root"])))
        if recorded - int((st.get("last_fit") or {}).get("recorded") or 0) < _student_need(st, cfg, now)[0]:
            return None
    return why


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
    if kind == STUDENT:
        p.update(student_params(lab, cfg))
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
            if str(params.get("tag") or "").startswith(STUDENT_TAG):  # a student candidate was judged: the loop's own bookkeeping
                return student_judged(lab, params)
        elif kind == STUDENT:
            return student_finish(lab, job_dir, params)
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


def _filler_basics(lab: Path, cfg: dict[str, Any], doc: dict, now: float, starting: bool, quiet_path: Path | str | None) -> str | None:
    """The rules every filler-like kind (teacher, student) shares: HALT or PAUSE up, a night running or due, a live QUIET.lock line that is
    not in the kind's `quiet_ignore`. The reason, or None."""
    from tools.lab import cores  # noqa: PLC0415 -- cores imports this module

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
    return f"QUIET.lock is held by {tag}" if tag else None


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
    lab = Path(lab)
    cfg = cfg if cfg is not None else config_from(read_lenient(lab / "jobs.json"))[TEACHER]
    now = time.time() if now is None else now
    from tools.lab import cores  # noqa: PLC0415 -- cores imports this module

    doc = cores.load_doc(lab)
    why = _filler_basics(lab, cfg, doc, now, starting, quiet_path)
    if why:
        return why
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


# ---------------------------------------------------------------- the student job (D56, D57 part 3, D58 C4; rl/run_student.py, tools/lab/student_job.py)
# The chain the runner closes (docs/lab/RUN_LOOP.md): `teacher` plays and labels; when 10,000 new labelled screens exist since the last fit (counted
# by `rl/run_student.py check`), or the weekly fit is due, `student` fits the run student, measures its legs, builds the candidate and writes
# `LAB/train/ready-<gen>-g<N>-student`; the `train` job judges it as it judges any candidate; a promotion rewrites agent/current.json, which the next
# teacher block reads. The stops (the human bar's freeze, `run-student`, `run-plateau`) and their reset live in LAB/student/.

STUDENT = "student"
STUDENT_TAG = "student"  # the run loop's candidates are `student`, `student-r2`, ...: markers `ready-<gen>-g<N>-student[-r<n>]`
STUDENT_DIR = "student"  # LAB/student/: state.json, frozen.json, verdicts.md
STUDENT_KINDS = ("card", "ancient", "rest", "route", "shop", "event", "reward", "select")  # rl/macro_recipe.STUDENT_KINDS (a test holds them equal)
HUMAN_BAR_KINDS = ("card", "ancient", "rest", "shop", "route", "event")  # rl/run_student.HUMAN_KINDS: the kinds the human record has choices for
FREEZE_AFTER = 2  # a kind that fails the human bar in this many rounds running is frozen
ALL_FAIL_AFTER = 2  # every kind failing the human bar in this many rounds running stops the student (LAB/alerts/run-student.txt)
PLATEAU_DAYS = 10.0  # no run-loop promotion in this many days writes LAB/alerts/run-plateau.txt
RUN_STUDENT_ALERT = "run-student"
RUN_PLATEAU_ALERT = "run-plateau"
ROUNDS_KEPT = 30
RUN_LOOP_FILE = "run-loop.md"  # LAB/run-loop.md: one line a day (teacher runs, screens, core-hours; the student's state)
RUN_LOOP_LINES = 90
STATUS_EVERY_S = 600.0
ROOT = Path(__file__).resolve().parents[2]
RUN_LEVEL_JSON = "D:/spireline/referee/human-bar/run-level.json"
_count_cache: dict[str, tuple[tuple, int | None, str]] = {}
_hb_module: Any = None


def student_dir(lab: Path | str) -> Path:
    return Path(lab) / STUDENT_DIR


def _fresh_state() -> dict[str, Any]:
    return {"version": 1, "loop_start": None, "round_count": 0, "last_fit": None, "rounds": [], "fails": {}, "all_fail": 0,
            "run_student_raised": False, "last_promotion": None, "plateau": {"raised": False, "ack": None}, "daily": None,
            "status_line": "", "status_t": 0.0}


def read_student_state(lab: Path | str) -> dict[str, Any]:
    """LAB/student/state.json over the defaults (a missing or damaged file is a fresh state)."""
    st = _fresh_state()
    doc = read_lenient(student_dir(lab) / "state.json")
    if isinstance(doc, dict):
        st.update({k: v for k, v in doc.items() if k in st})
    if not isinstance(st["rounds"], list):
        st["rounds"] = []
    if not isinstance(st["fails"], dict):
        st["fails"] = {}
    if not isinstance(st["plateau"], dict):
        st["plateau"] = {"raised": False, "ack": None}
    return st


def write_student_state(lab: Path | str, st: dict[str, Any]) -> None:
    _atomic(student_dir(lab) / "state.json", json.dumps(st, indent=1) + "\n")


def read_frozen(lab: Path | str) -> dict[str, dict[str, Any]]:
    """The kinds the student may not take: LAB/student/frozen.json `kinds` ({kind: {since, round, failures, why}}). The orchestrator unfreezes a kind
    by deleting its entry (or the file); the failure count then starts again (`update_human_bar`)."""
    doc = read_lenient(student_dir(lab) / "frozen.json")
    kinds = doc.get("kinds") if isinstance(doc, dict) else None
    if not isinstance(kinds, dict):
        return {}
    return {k: (v if isinstance(v, dict) else {}) for k, v in kinds.items() if k in STUDENT_KINDS}


def write_frozen(lab: Path | str, frozen: dict[str, dict[str, Any]]) -> None:
    note = ("A kind listed here failed the human bar (leg 2 of rl/run_student.py) in two rounds running; the student job leaves it out of the fit "
            "and the candidate. Delete its entry (or this file) to let it be fitted again: its failure count then starts over.")
    _atomic(student_dir(lab) / "frozen.json", json.dumps({"_note": note, "kinds": frozen}, indent=1) + "\n")


def _iso(t: float) -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(t))


# ---- counting the labelled screens

def student_shard_dirs(root: Path | str) -> list[Path]:
    """The teacher's decision shards under `root` (the daily job's `<generation>/shards/<job>` and a hand-started round's `<store>/<generation>/shards/<job>`),
    found by two fixed-depth globs: the store holds thousands of per-run files that a recursive walk would visit at every pick."""
    root = Path(root)
    out: list[Path] = []
    for pat in ("*/shards/*", "*/*/shards/*"):
        out += [d for d in sorted(root.glob(pat)) if (d / "decisions.jsonl").is_file()]
    return out


def student_recorded_screens(root: Path | str) -> tuple[int, tuple]:
    """(screens recorded in every shard's manifest, a signature of the shards): the cheap count, an upper bound on what `check` keeps."""
    total = 0
    sig: list[tuple[str, int, int]] = []
    for d in student_shard_dirs(root):
        f = d / "decisions.jsonl"
        try:
            stt = f.stat()
        except OSError:
            continue
        man = read_lenient(d / "manifest.json")
        n = ((man.get("counts") or {}).get("decisions") if isinstance(man, dict) else None)
        if not isinstance(n, int) or isinstance(n, bool):
            n = f.read_bytes().count(b"\n")
        total += n
        sig.append((str(d), stt.st_size, stt.st_mtime_ns))
    return total, tuple(sig)


JOB_TREE: Path | None = None  # the lab tree (the runner's --tree) the student job runs in; the runner sets it (`use_tree`); None: this module's own checkout


def use_tree(tree: Path | str | None) -> None:
    """The lab tree the jobs run in, for the parts of this module that need a file of it before a job exists (the runner calls this once, at its start)."""
    global JOB_TREE
    JOB_TREE = None if tree is None else Path(tree)


def job_tree(cfg: dict[str, Any]) -> Path:
    """The folder the student job runs in: the entry's `cwd` (default `{tree}`) with `{tree}` as the runner resolves it (its --tree). A `cwd` that names
    another token (`{lab}`) or none of this module's business falls back to the tree itself; with no tree set, this module's own checkout."""
    tree = JOB_TREE if JOB_TREE is not None else ROOT
    cwd = str(cfg.get("cwd") or "{tree}").replace("\\", "/")
    got = cwd.replace("{tree}", tree.as_posix())
    return tree if "{" in got else Path(got)


def student_check_cmd(cfg: dict[str, Any], root: Path | str) -> list[str]:
    """The command that counts what the student would learn from: `rl/run_student.py check --shards-root <root>`, run from the tree the job itself runs in
    (`job_tree`: the lab tree, whose `out/` a presort rebuilds; the runner's own checkout is not rebuilt and its library goes stale at every engine landing),
    in the runner's own Python. `check_cmd` in the entry replaces it, with `{root}` and `{python}`."""
    import sys  # noqa: PLC0415

    cmd = cfg.get("check_cmd")
    if isinstance(cmd, (list, tuple)) and cmd:
        return [str(t).replace("{root}", str(root)).replace("{python}", sys.executable) for t in cmd]
    return [sys.executable, str(job_tree(cfg) / "rl" / "run_student.py"), "check", "--shards-root", str(root)]


def student_exact_screens(root: Path | str, cfg: dict[str, Any]) -> tuple[int | None, str]:
    """(decisions the student's loader keeps, why not) from `check`: its `decisions`. None, with a reason, when the command fails or prints no count."""
    import subprocess  # noqa: PLC0415

    try:
        r = subprocess.run(student_check_cmd(cfg, root), capture_output=True, text=True, check=False, timeout=float(cfg.get("check_timeout_s", 900)),
                           env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    except (OSError, subprocess.SubprocessError) as e:
        return None, f"the check could not run ({type(e).__name__})"
    if r.returncode != 0:
        from tools.lab import alerts  # noqa: PLC0415

        last = [ln for ln in r.stderr.splitlines() if ln.strip()][-1:] or ["no message"]
        return None, f"the check failed (rc {r.returncode}): {alerts.scrub(last[0])}"
    try:
        doc = json.loads(r.stdout)
        n = doc["decisions"]
    except (ValueError, KeyError, TypeError):
        return None, "the check printed no count"
    return (int(n), "") if isinstance(n, int) and not isinstance(n, bool) else (None, "the check printed no count")


def _exact_cached(root: Path | str, cfg: dict[str, Any], sig: tuple) -> tuple[int | None, str]:
    """`student_exact_screens`, asked again only when a shard changed (a pick reads this every few minutes; `check` loads every row)."""
    key = str(root)
    got = _count_cache.get(key)
    if got is not None and got[0] == sig:
        return got[1], got[2]
    kept, why = student_exact_screens(root, cfg)
    _count_cache[key] = (sig, kept, why)
    return kept, why


LABELLED_SHORT = "new labelled screens since the last fit"  # student_due's reason when `check` keeps too few: the kind is not due, whatever holds it


def _student_need(st: dict[str, Any], cfg: dict[str, Any], now: float) -> tuple[int, int, int, bool]:
    """(new screens the trigger wants now, the threshold, the weekly floor, whether the weekly fit is due)."""
    last = st.get("last_fit") or {}
    threshold, floor_ = int(cfg.get("new_screens", 10000)), int(cfg.get("weekly_min_screens", 1000))
    since = last.get("t") or st.get("loop_start")
    weekly = since is not None and now - float(since) >= float(cfg.get("weekly_days", 7.0)) * 86400
    return (floor_ if weekly else threshold), threshold, floor_, weekly


def student_due(st: dict[str, Any], cfg: dict[str, Any], root: Path | str, now: float) -> tuple[bool, str, int | None, int]:
    """(due, reason, labelled screens `check` keeps, screens recorded). Due when `new_screens` (10,000) new labelled screens exist since the last fit, or
    when the weekly fit is due (`weekly_days` since the last fit, or since the loop started when there is none) and `weekly_min_screens` are new."""
    recorded, sig = student_recorded_screens(root)
    last = st.get("last_fit") or {}
    need, threshold, floor_, weekly = _student_need(st, cfg, now)
    new_rec = recorded - int(last.get("recorded") or 0)
    if new_rec < need:
        return False, (f"{max(0, new_rec)} new screens recorded since the last fit; the student wants {threshold}"
                       f" (or {floor_} once the weekly fit is due)"), None, recorded
    kept, why = _exact_cached(root, cfg, sig)
    if kept is None:
        return False, f"cannot count the labelled screens: {why}", None, recorded
    new_kept = kept - int(last.get("screens") or 0)
    if new_kept < need:
        return False, f"{max(0, new_kept)} {LABELLED_SHORT} (check); the student wants {need}", kept, recorded
    return True, f"{new_kept} new labelled screens" + (" (the weekly fit)" if weekly and new_kept < threshold else ""), kept, recorded


# ---- the incumbent, the tag, the hold and the params

def student_base(cfg: dict[str, Any]) -> dict[str, Any] | str:
    """The incumbent the student starts from (agent_root/current.json): its generation, folder and the prior the fit starts from (its student's
    file when it has one, else its human prior's), and `teacher_prior`: the human prior the teacher decided with (rl/run_teacher.py pulls by pi_h), named
    only when the fit starts from the student's file, which is not it (the loader refuses a store whose recorded log-probabilities are not the
    prior's it is given); None when the start prior is the human prior itself or the incumbent names none. The reason as a string when there is none."""
    root = Path(str(cfg.get("agent_root") or DEFAULTS[STUDENT]["agent_root"]))
    cur = read_lenient(root / "current.json")
    if not isinstance(cur, dict) or not isinstance(cur.get("generation"), int) or not cur.get("dir"):
        return "agent/current.json is not readable"
    d = Path(str(cur["dir"]).replace("\\", "/"))
    if not d.is_absolute():
        d = root / d
    meta = read_lenient(d / "agent.json")
    if not isinstance(meta, dict):
        return f"{d.name}/agent.json is not readable"
    stu, hum = meta.get("student"), meta.get("human_prior")
    teacher_prior: Path | None = None
    if isinstance(stu, dict):
        prior, what = d / str(stu.get("file") or "run_student.npz"), "student"
        if isinstance(hum, dict):
            teacher_prior = d / str(hum.get("file") or "human_prior.npz")
    elif isinstance(hum, dict):
        prior, what = d / str(hum.get("file") or "human_prior.npz"), "human prior"
    else:
        return "the incumbent plays no human prior or student to start the fit from"
    for f in (prior, teacher_prior):
        if f is not None and not f.is_file():
            return f"{f.name} is missing in {d.name}"
    return {"generation": int(cur["generation"]), "dir": d, "prior": prior, "prior_kind": what, "root": root, "teacher_prior": teacher_prior}


def student_tag(lab: Path | str, agent_root: Path | str, generation: int, target: int) -> str:
    """`student`, else `student-r2`, `student-r3`, ...: the first tag with no candidate folder and no marker (ready or done) for it at this target."""
    n = 1
    while True:
        tag = STUDENT_TAG if n == 1 else f"{STUDENT_TAG}-r{n}"
        taken = (Path(agent_root) / f"gen{target}" / f"candidate-{tag}").exists() or any(
            (Path(lab) / "train" / f"{pre}-{generation}-g{target}-{tag}").exists() for pre in ("ready", "done"))
        if not taken:
            return tag
        n += 1


def student_hold(lab: Path | str, cfg: dict[str, Any] | None = None, *, starting: bool = True, now: float | None = None,
                 quiet_path: Path | str | None = None, state: dict[str, Any] | None = None) -> str | None:
    """Why the student job may not start; None when it may. The filler's rules (HALT, PAUSE, the night, a live QUIET.lock line outside `quiet_ignore`),
    judgement first (a waiting train marker: one candidate is judged at a time), the stop (LAB/alerts/run-student.txt), a usable incumbent and the
    trigger: 10,000 new labelled screens since the last fit, or the weekly fit with at least `weekly_min_screens` new."""
    from tools.lab import alerts, cores  # noqa: PLC0415 -- cores imports this module

    lab = Path(lab)
    cfg = cfg if cfg is not None else config_from(read_lenient(lab / "jobs.json"))[STUDENT]
    now = time.time() if now is None else now
    if alerts.alert_path(lab, RUN_STUDENT_ALERT).exists():
        return (f"stopped: every kind failed the human bar twice running (LAB/alerts/{RUN_STUDENT_ALERT}.txt); "
                "remove the file by hand to let it run again")
    why = _filler_basics(lab, cfg, cores.load_doc(lab), now, starting, quiet_path)
    if why:
        return why
    marks = train_markers(lab)
    if marks:
        return f"judgement first: {marks[0].name} is waiting"
    base = student_base(cfg)
    if isinstance(base, str):
        return base
    st = state if state is not None else read_student_state(lab)
    root = Path(str(cfg.get("shards_root") or DEFAULTS[STUDENT]["shards_root"]))
    due, reason, _, _ = student_due(st, cfg, root, now)
    return None if due else reason


def student_params(lab: Path | str, cfg: dict[str, Any]) -> dict[str, Any]:
    """The job's own parameters, fixed when the job is made (a resumed job keeps them): the command with `{student_prior}`, `{student_teacher_prior}`,
    `{student_target}` and `{student_tag}` filled (when the fit starts from the human prior itself there is no teacher prior: the token and the flag
    before it are left out), the incumbent's generation, the candidate's number and tag, its marker, the screens the fit starts from."""
    base = student_base(cfg)
    if isinstance(base, str):
        raise ValueError(base)
    gen, target = base["generation"], base["generation"] + 1
    tag = student_tag(lab, base["root"], gen, target)
    root = Path(str(cfg.get("shards_root") or DEFAULTS[STUDENT]["shards_root"]))
    recorded, sig = student_recorded_screens(root)
    kept, _ = _exact_cached(root, cfg, sig) if recorded else (0, "")
    tp = base.get("teacher_prior")
    cmd: list[str] = []
    for t in (str(x) for x in cfg["cmd"]):
        if "{student_teacher_prior}" in t and tp is None:  # nothing to name: the flag that takes it goes too (an element that is only the flag)
            if t == "{student_teacher_prior}" and cmd and cmd[-1].startswith("--"):
                cmd.pop()
            continue
        cmd.append(t.replace("{student_prior}", base["prior"].as_posix()).replace("{student_teacher_prior}", "" if tp is None else tp.as_posix())
                   .replace("{student_target}", str(target)).replace("{student_tag}", tag))
    return {"cmd": cmd, "generation": str(gen), "candidate_generation": target, "tag": tag, "marker": f"ready-{gen}-g{target}-{tag}",
            "ok_rcs": [int(x) for x in (cfg.get("ok_rcs") or [0, 3])], "screens": kept, "recorded": recorded, "prior_kind": base["prior_kind"]}


# ---- the human bar's stops

def update_human_bar(st: dict[str, Any], frozen: dict[str, dict[str, Any]], verdict: dict[str, Any], round_no: int, now: float) -> list[str]:
    """One round's legs into the stops. A kind with a human fit whose leg 2 failed counts a failure (a pass resets it; a kind with no measurement is
    left as it was); two in a row freeze the kind (`frozen`, edited in place; the caller writes LAB/student/frozen.json). When every measured kind failed
    this round and the one before, `all_fail` reaches 2 (the caller raises LAB/alerts/run-student.txt). A kind whose count stands at the limit but which
    is not in `frozen` was unfrozen by hand: its count starts again. Returns the notes for the log."""
    notes: list[str] = []
    fails: dict[str, int] = st["fails"]
    for k in list(fails):
        if int(fails[k]) >= FREEZE_AFTER and k not in frozen:
            fails[k] = 0
            notes.append(f"{k}: unfrozen by hand, its failure count starts again")
    measured: dict[str, bool] = {}
    for k in HUMAN_BAR_KINDS:
        v = verdict.get(k)
        if k in frozen or not isinstance(v, dict) or v.get("leg2") not in (True, False) or v.get("leg2_measured") is False:
            continue  # (lane 3 reports an unmeasured human bar as a failed one, `leg2` False: the job says `leg2_measured` False, and that is no failure of the student)
        measured[k] = v["leg2"] is True
    for k, ok in measured.items():
        fails[k] = 0 if ok else int(fails.get(k, 0)) + 1
        if fails[k] >= FREEZE_AFTER and k not in frozen:
            frozen[k] = {"since": _iso(now), "round": round_no, "failures": fails[k],
                         "why": f"failed the human bar (leg 2) in {FREEZE_AFTER} rounds running"}
            notes.append(f"{k}: frozen after {fails[k]} rounds failing the human bar (LAB/student/frozen.json)")
    st["all_fail"] = int(st.get("all_fail") or 0) + 1 if measured and not any(measured.values()) else 0
    return notes


def _hb():
    """rl/run_human_bar.py by path (the runner has no `rl` on its path; the module imports only the standard library at its top)."""
    global _hb_module
    if _hb_module is None:
        import importlib.util  # noqa: PLC0415

        spec = importlib.util.spec_from_file_location("run_human_bar_lab", ROOT / "rl" / "run_human_bar.py")
        mod = importlib.util.module_from_spec(spec)  # type: ignore[arg-type]
        spec.loader.exec_module(mod)  # type: ignore[union-attr]
        _hb_module = mod
    return _hb_module


def _verdicts_append(lab: Path, text: str) -> None:
    p = student_dir(lab) / "verdicts.md"
    p.parent.mkdir(parents=True, exist_ok=True)
    old = p.read_text(encoding="utf-8") if p.exists() else "# The run loop's verdicts (newest last; the run-level human bar beside each)\n"
    _atomic(p, old.rstrip("\n") + "\n\n" + text.rstrip("\n") + "\n")


def _round_table(verdict: dict[str, Any]) -> str:
    rows = ["| kind | leg 1 (teacher) | human bar (leg 2) | passes | why not |", "|---|---|---|---|---|"]
    word = {True: "pass", False: "FAIL", None: "no human fit"}
    for k in STUDENT_KINDS:
        v = verdict.get(k)
        if isinstance(v, dict):
            leg2 = "not measured" if v.get("leg2_measured") is False else word.get(v.get("leg2"), "-")
            rows.append(f"| {k} | {word.get(v.get('leg1'), '-')} | {leg2} | {'yes' if v.get('pass') else 'no'} | {v.get('why') or ''} |")
    return "\n".join(rows)


def student_finish(lab: Path | str, job_dir: Path | str, params: dict[str, Any], *, now: float | None = None) -> str | None:
    """After a student job ran (rc 0, or 3: no kind passed): the round into LAB/student/state.json, the human bar's stops (freeze, `run-student`),
    the marker `ready-<gen>-g<N>-<tag>` when a candidate was built, the verdict with the run-level bar beside it. A problem string or None; the
    bookkeeping never fails the job that did the work."""
    try:
        return _student_finish(Path(lab), Path(job_dir), params, time.time() if now is None else now)
    except Exception as e:  # noqa: BLE001
        return f"student: bookkeeping failed ({type(e).__name__})"


def _student_finish(lab: Path, job_dir: Path, params: dict[str, Any], now: float) -> str | None:
    from tools.lab import alerts  # noqa: PLC0415

    cfg = config_from(read_lenient(lab / "jobs.json"))[STUDENT]
    res = read_result(job_dir)
    st, frozen = read_student_state(lab), read_frozen(lab)
    verdict = res.get("verdict") if isinstance(res.get("verdict"), dict) else {}
    st["round_count"] = int(st.get("round_count") or 0) + 1
    n = st["round_count"]
    notes = update_human_bar(st, frozen, verdict, n, now)
    kept = res.get("decisions") if isinstance(res.get("decisions"), int) else params.get("screens")
    st["last_fit"] = {"t": now, "utc": _iso(now), "round": n, "tag": params.get("tag"), "screens": int(kept or 0),
                      "recorded": int(params.get("recorded") or 0), "built": bool(res.get("built"))}
    marker = None
    cand = res.get("candidate") if res.get("built") else None
    if cand and (Path(str(cfg["agent_root"])) / str(cand) / "agent.json").is_file() and params.get("marker"):
        marker = str(params["marker"])
        if not (lab / "train" / marker).exists():
            _atomic(lab / "train" / marker, _iso(now) + "\n")
    entry = {"round": n, "utc": _iso(now), "tag": params.get("tag"), "generation": params.get("generation"), "target": params.get("candidate_generation"),
             "screens": int(kept or 0), "built": bool(marker), "candidate": cand if marker else None, "marker": marker,
             "kinds_passing": sorted(res.get("kinds_passing") or {}),
             "verdict": {k: {key: v.get(key) for key in ("pass", "leg1", "leg2", "leg2_measured", "why")} for k, v in verdict.items() if isinstance(v, dict)},
             "frozen": sorted(frozen), "judged": None}
    st["rounds"] = [*st["rounds"], entry][-ROUNDS_KEPT:]
    if notes or frozen != read_frozen(lab):
        write_frozen(lab, frozen)
    raised = False
    if int(st["all_fail"]) >= ALL_FAIL_AFTER and not alerts.alert_path(lab, RUN_STUDENT_ALERT).exists():
        alerts.write_run_student(lab, [r for r in st["rounds"] if r.get("verdict")][-ALL_FAIL_AFTER:], now=now)
        st["run_student_raised"] = raised = True
    write_student_state(lab, st)
    head = (f"## {_iso(now)} student round {n} ({params.get('tag')}, incumbent generation {params.get('generation')}): "
            + (f"candidate {cand}, marker {marker}" if marker else "no candidate (no kind passed both legs with a measured lambda)"))
    parts = [head, "", f"Fit on {int(kept or 0):,} labelled screens ({int(params.get('recorded') or 0):,} recorded); the fit started from the incumbent's "
             f"{params.get('prior_kind', 'prior')}.", "", _round_table(verdict)]
    if notes:
        parts += ["", *[f"- {x}" for x in notes]]
    if raised:
        parts += ["", f"- every kind with a human bar failed it in {ALL_FAIL_AFTER} rounds running: LAB/alerts/{RUN_STUDENT_ALERT}.txt written, the student is held"]
    bar = job_dir / "run-bar.md"
    parts += ["", bar.read_text(encoding="utf-8") if bar.is_file() else "run-level human bar: no recorded runs of the copy yet"]
    _verdicts_append(lab, "\n".join(parts))
    return None


def student_judged(lab: Path | str, params: dict[str, Any], *, now: float | None = None) -> str | None:
    """After the train job judged a student candidate (its marker was consumed): the verdict from the candidate's promotion.json into the round, a
    promotion into `last_promotion` (the plateau's clock), and the verdict with the run-level bar beside it, from the candidate's arms."""
    try:
        return _student_judged(Path(lab), params, time.time() if now is None else now)
    except Exception as e:  # noqa: BLE001
        return f"student: bookkeeping failed ({type(e).__name__})"


def _student_judged(lab: Path, params: dict[str, Any], now: float) -> str | None:
    cfg = config_from(read_lenient(lab / "jobs.json"))[STUDENT]
    cdir = Path(str(cfg["agent_root"])) / f"gen{params.get('candidate_generation')}" / f"candidate-{params.get('tag')}"
    promo = read_lenient(cdir / "promotion.json")
    dec = promo.get("decision") if isinstance(promo, dict) else None
    promoted = bool(isinstance(dec, dict) and dec.get("promote") is True)
    verdict = "promoted" if promoted else "rejected" if isinstance(dec, dict) else "no verdict"  # no promotion.json: the round ended before a verdict
    st = read_student_state(lab)
    for entry in reversed(st["rounds"]):
        if entry.get("tag") == params.get("tag") and entry.get("target") == params.get("candidate_generation"):
            entry["judged"] = verdict
            break
    if promoted:
        st["last_promotion"] = now
    write_student_state(lab, st)
    hb = _hb()
    copy = hb.copy_from_arms(sorted(cdir.glob("arms/*/runs.jsonl")))
    human = hb.read_human(cfg.get("run_level") or RUN_LEVEL_JSON)
    rule = (dec or {}).get("rule") if isinstance(dec, dict) else None
    said = {"promoted": "PROMOTED", "rejected": "not promoted"}.get(verdict, "no verdict (the round ended without a promotion.json)")
    _verdicts_append(lab, f"## {_iso(now)} judgement of {params.get('marker')}: {said}"
                     + (f" (rule {rule})" if rule else "") + "\n\n" + hb.beside_text(human, copy, label="candidate"))
    return None


# ---- the daily line, the plateau, the status

def teacher_day_figures(store: Path | str, day: str) -> dict[str, Any]:
    """The teacher's runs that ended on the UTC day `day`, from the ledger of the store and of the stores beside it (a hand-started round 0): the runs,
    the screens they labelled and the CPU they used in core-hours."""
    store = Path(store)
    runs = screens = 0
    cpu = 0.0
    for f in [store / "ledger.jsonl", *sorted(store.glob("*/ledger.jsonl"))]:
        for r in read_ledger(f):
            if r.get("ev") != "end" or r.get("retracted") or str(r.get("t", ""))[:10] != day:
                continue
            runs += 1
            screens += int(r.get("screens") or 0) if isinstance(r.get("screens"), (int, float)) else 0
            cpu += float(r.get("cpu") or 0.0) if isinstance(r.get("cpu"), (int, float)) else 0.0
    return {"runs": runs, "screens": screens, "core_hours": round(cpu / 3600.0, 2)}


def run_loop_line(day: str, cfgs: dict[str, dict[str, Any]], st: dict[str, Any], frozen: dict[str, Any]) -> str:
    """One day's line: the teacher's runs, screens and core-hours, then the student's state."""
    t = teacher_day_figures(cfgs[TEACHER].get("store") or "D:/spireline/referee/run-teacher", day)
    last = st.get("last_fit") or {}
    promoted = st.get("last_promotion")
    return (f"{day} teacher: {t['runs']} runs, {t['screens']:,} screens, {t['core_hours']:.1f} core-hours (budget "
            f"{float(cfgs[TEACHER].get('core_hours_per_day', 24.0)):g}/day); student: {int(st.get('round_count') or 0)} rounds"
            + (f", last fit {str(last.get('utc', ''))[:10]} on {int(last.get('screens') or 0):,} screens" if last else ", no fit yet")
            + f", frozen {','.join(sorted(frozen)) or '-'}; last run-loop promotion "
            + (_iso(float(promoted))[:10] if isinstance(promoted, (int, float)) else "none"))


def _append_daily(lab: Path, day: str, line: str) -> None:
    p = Path(lab) / RUN_LOOP_FILE
    head = "# The run loop, one line a day (LAB/student/ holds the state; tools/lab/jobs.py run_loop_tick writes this)\n"
    lines = p.read_text(encoding="utf-8").splitlines() if p.exists() else []
    body = [ln for ln in lines if ln and not ln.startswith("#")]
    if any(ln.startswith(day + " ") for ln in body):
        return
    body = [*body, line][-RUN_LOOP_LINES:]
    _atomic(p, head + "\n".join(body) + "\n")


def run_loop_section(lab: Path | str, lines: int = 7) -> str:
    """The last `lines` days of LAB/run-loop.md as a section for clock.md ("" when there is none)."""
    p = Path(lab) / RUN_LOOP_FILE
    try:
        body = [ln for ln in p.read_text(encoding="utf-8").splitlines() if ln and not ln.startswith("#")]
    except OSError:
        return ""
    return ("\n## The run loop (LAB/run-loop.md)\n\n" + "\n".join(f"- {ln}" for ln in body[-lines:]) + "\n") if body else ""


def run_loop_status(lab: Path | str) -> str:
    """The short text status.txt carries after its line ("" when the loop is off): today's teacher figures and the student's state."""
    return str(read_student_state(lab).get("status_line") or "")


def run_loop_tick(lab: Path | str, now: float | None = None) -> list[str]:
    """What the runner does about the loop between jobs (each pass and after a teacher, student or train job): start the loop's clock the first time
    it is on; an alert file removed by hand starts its count again; raise `run-plateau` after PLATEAU_DAYS without a run-loop promotion; write the
    closed day's line once a day; refresh the status text. Returns the log lines. Does nothing while neither the teacher nor the student is on."""
    from tools.lab import alerts  # noqa: PLC0415

    lab = Path(lab)
    now = time.time() if now is None else now
    cfgs = config_from(read_lenient(lab / "jobs.json"))
    if not (cfgs[STUDENT].get("enabled") or cfgs[TEACHER].get("enabled")):
        return []
    st, frozen = read_student_state(lab), read_frozen(lab)
    before = json.dumps(st, sort_keys=True)
    msgs: list[str] = []
    if st["loop_start"] is None:
        st["loop_start"] = now
    if st.get("run_student_raised") and not alerts.alert_path(lab, RUN_STUDENT_ALERT).exists():
        st["run_student_raised"], st["all_fail"] = False, 0
        msgs.append(f"run-loop: {RUN_STUDENT_ALERT}.txt removed by hand: the student runs again and the count of failing rounds starts over")
    pl = st["plateau"]
    path = alerts.alert_path(lab, RUN_PLATEAU_ALERT)
    if pl.get("raised") and not path.exists():
        pl["raised"], pl["ack"] = False, now
        msgs.append(f"run-loop: {RUN_PLATEAU_ALERT}.txt removed by hand: the {PLATEAU_DAYS:g} days count again from now")
    if path.exists():
        pl["raised"] = True
    else:
        anchor = max(float(st.get("last_promotion") or 0.0), float(st["loop_start"]), float(pl.get("ack") or 0.0))
        if now - anchor >= PLATEAU_DAYS * 86400:
            alerts.write_run_plateau(lab, anchor, PLATEAU_DAYS, st["rounds"][-3:], now=now)
            pl["raised"] = True
            msgs.append(f"ALERT {RUN_PLATEAU_ALERT}: no run-loop promotion since {_iso(anchor)[:10]} ({PLATEAU_DAYS:g} days); LAB/alerts/{RUN_PLATEAU_ALERT}.txt")
    today = _iso(now)[:10]
    if st.get("daily") != today:
        yesterday = _iso(now - 86400)[:10]
        line = run_loop_line(yesterday, cfgs, st, frozen)
        _append_daily(lab, yesterday, line)
        st["daily"] = today
        msgs.append("run-loop: " + line)
    if now - float(st.get("status_t") or 0.0) >= STATUS_EVERY_S:
        t = teacher_day_figures(cfgs[TEACHER].get("store") or "D:/spireline/referee/run-teacher", today)
        st["status_line"] = (f"run-loop today: teacher {t['runs']} runs {t['screens']} screens {t['core_hours']:.1f} core-h; "
                             f"student {int(st.get('round_count') or 0)} rounds, frozen {','.join(sorted(frozen)) or '-'}")
        st["status_t"] = now
    after = json.dumps(st, sort_keys=True)
    if after != before:
        write_student_state(lab, st)
    return msgs
