"""The agent-era job kinds of the CPU schedule (docs/PLAN.md section 7): selfplay, train, score-real, bench-env, bench-emu.

Each kind is data: a default command, a limit, whether it is on, and the input that makes it ready. ``LAB/jobs.json``
(optional, read at every choice, keys starting with "_" are notes) overrides any of it per kind:

  {"selfplay": {"enabled": true, "cmd": ["{python}", "rl/run_train/cli.py", "..."], "share": 0.5, "timeout_min": 60}}

The shipped defaults change nothing for a lab that has no jobs.json: selfplay and the benches are off, and train and
score-real are on but only ever ready when their marker file exists (LAB/train/ready-<gen>, ready-<gen>-r<n> for a retrain after a rejection or ready-<gen>-g<target> for a candidate that is generation <target>, LAB/score/pending).

Priority ladder (lower number first; what the runner's choose() does with it is in runner.py):

  1 presort   2 score-real   3 train   4 aimed fights   5 runs / rank   6 selfplay   7 the rest, bench-* nightly

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

NEW_KINDS = ("selfplay", "train", "score-real", "bench-env", "bench-emu")
BENCHES = ("bench-env", "bench-emu")
# The kinds that exist only because self-play does: self-play, the training (and the promotion it runs) of what it recorded, and the
# real-game score of what training promoted. `selfplay.share` is judged on their CPU together (mix.loop_cpu), not on self-play alone.
LOOP_KINDS = ("selfplay", "train", "score-real")
PRIORITY = {"presort": 1, "score-real": 2, "train": 3, "fights-aimed": 4, "runs": 5, "selfplay": 6, "rest": 7,
            "bench-env": 7, "bench-emu": 7}
SCORE_WINDOW_H = 4.0  # the longest a score-real window may run
ACTIVE_STALE_H = SCORE_WINDOW_H + 1.0  # an `active` marker older than this is a crash's leftover


def _placeholder(kind: str, entry: str) -> list[str]:
    """A command that only prints what it stands for; A2 sets the real one in LAB/jobs.json."""
    return ["{python}", "-c", f"print('placeholder: set LAB/jobs.json {kind}.cmd; entry point {entry}')"]


DEFAULTS: dict[str, dict[str, Any]] = {
    "selfplay": {"enabled": False, "cmd": _placeholder("selfplay", "rl/run_train/cli.py"), "cwd": "{tree}", "timeout_min": 60,
                 "share": 0.0, "needs": []},
    "train": {"enabled": True, "cmd": _placeholder("train", "rl/run_train/cli.py"), "cwd": "{tree}", "timeout_min": 60, "gpu": True,
              "needs": []},
    "score-real": {"enabled": True, "cmd": _placeholder("score-real", "rl/score_real.py"), "cwd": "{tree}", "timeout_min": 240,
                   "procs": 14, "weekly": False, "weekly_days": 7.0, "needs": []},
    "bench-env": {"enabled": False, "cmd": _placeholder("bench-env", "the envproto bench on branch env/leak-probe"), "cwd": "{tree}",
                  "timeout_min": 40, "procs": 12, "every_h": 20.0, "needs": []},
    "bench-emu": {"enabled": False, "cmd": ["{python}", "rl/bench_emu.py"], "cwd": "{tree}", "timeout_min": 30, "every_h": 20.0,
                  "needs": ["{tree}/rl/bench_emu.py"]},
}
BENCH_KEY = {"bench-env": "env", "bench-emu": "emu"}


def read_lenient(p: Path) -> Any:
    try:
        return json.loads(p.read_bytes().decode("utf-8", errors="replace").lstrip("﻿").replace("\x00", ""))
    except (OSError, ValueError):
        return None


def load_config(lab: Path) -> dict[str, dict[str, Any]]:
    """The settings of every new kind: the defaults, with LAB/jobs.json's per-kind keys laid over them."""
    cfg = {k: dict(v) for k, v in DEFAULTS.items()}
    doc = read_lenient(Path(lab) / "jobs.json")
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


def gen_key(name: str) -> tuple[int, int, int, str]:
    """Marker order: by generation, then the plain and retrain forms by index, then the target forms by target
    (``ready-4`` < ``ready-4-r1`` < ``ready-4-g6`` < ``ready-12``)."""
    m = re.search(r"-(\d+)(?:-r(\d+)|-g(\d+))?$", name)
    if not m:
        return (1 << 30, 0, 0, name)
    return (int(m.group(1)), 1 if m.group(3) else 0, int(m.group(3) or m.group(2) or 0), name)


def train_markers(lab: Path) -> list[Path]:
    d = Path(lab) / "train"
    return sorted(d.glob("ready-*"), key=lambda p: gen_key(p.name)) if d.is_dir() else []


def score_active(lab: Path) -> bool:
    """A score-real window is open (its marker is fresh): fights jobs must not start."""
    a = _age_h(Path(lab) / "score" / "active")
    return a is not None and a < ACTIVE_STALE_H


def due(lab: Path, kind: str, every_h: float) -> bool:
    a = _age_h(Path(lab) / kind / "last-ok")
    return a is None or a >= every_h


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
    if kind in BENCHES:
        return None if due(lab, kind, float(cfg.get("every_h", 20.0))) else "ran within the last night"
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
        target = re.fullmatch(r"ready-(\d+)-g(\d+)", p["marker"])
        if target:  # `ready-<gen>-g<target>`: the shards of <gen> train a candidate that is generation <target> (D33 addendum (b))
            p["generation"], p["candidate_generation"] = target.group(1), int(target.group(2))
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


def finish(kind: str, lab: Path, job_dir: Path, params: dict[str, Any]) -> str | None:
    """After the command succeeded: consume the input marker, stamp the night, store the bench. A problem string or None."""
    lab = Path(lab)
    stamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()) + "\n"
    try:
        if kind == "train" and params.get("marker"):
            m = lab / "train" / params["marker"]
            if m.exists():
                os.replace(m, m.with_name("done-" + params["marker"].removeprefix("ready-")))
        elif kind == "score-real":
            (lab / "score" / "pending").unlink(missing_ok=True)
            _atomic(lab / "score" / "last-ok", stamp)
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
