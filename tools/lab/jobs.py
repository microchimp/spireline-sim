"""The agent-era job kinds of the CPU schedule (docs/PLAN.md section 7): selfplay, train, score-real, nightly-real, bench-env, bench-emu.

Each kind is data: a default command, a limit, whether it is on, and the input that makes it ready. ``LAB/jobs.json``
(optional, read at every choice, keys starting with "_" are notes) overrides any of it per kind:

  {"selfplay": {"enabled": true, "cmd": ["{python}", "rl/run_train/cli.py", "..."], "share": 0.5, "timeout_min": 60}}

The shipped defaults change nothing for a lab that has no jobs.json: selfplay and the benches are off, and train and
score-real are on but only ever ready when their marker file exists (LAB/train/ready-<gen>, ready-<gen>-r<n> for a retrain after a rejection or ready-<gen>-g<target> for a candidate that is generation <target>, LAB/score/pending).

Priority ladder (lower number first; what the runner's choose() does with it is in runner.py):

  1 presort   2 score-real, nightly-real   3 train   4 aimed fights   5 runs / rank   6 selfplay   7 the rest, bench-* nightly

nightly-real is a kind of its own (D8, D33 addendum 2): score-real is one command on one marker, the nightly block is a driver
(rl/nightly_real.py) that plays several agents under the QUIET.lock, so the due rule and the window markers sit here and the
work sits in the driver. Due: LAB/nightly/last-ok older than `every_h` (20); skipped, with the reason, while LAB/HALT or
LAB/PAUSE is up, LAB/nightly/active is fresh (the last block has not finished) or LAB/score/active is fresh.

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

NEW_KINDS = ("selfplay", "train", "score-real", "nightly-real", "bench-env", "bench-emu")
BENCHES = ("bench-env", "bench-emu")
# The kinds that exist only because self-play does: self-play, the training (and the promotion it runs) of what it recorded, and the
# real-game score of what training promoted. `selfplay.share` is judged on their CPU together (mix.loop_cpu), not on self-play alone.
LOOP_KINDS = ("selfplay", "train", "score-real")  # nightly-real is outside the loop share: it scores the incumbent whether or not self-play ran (D29 accounting)
PRIORITY = {"presort": 1, "score-real": 2, "nightly-real": 2, "train": 3, "fights-aimed": 4, "runs": 5, "selfplay": 6, "rest": 7,
            "bench-env": 7, "bench-emu": 7}
SCORE_WINDOW_H = 4.0  # the longest a score-real window may run
ACTIVE_STALE_H = SCORE_WINDOW_H + 1.0  # an `active` marker older than this is a crash's leftover
NIGHTLY_WINDOW_H = 10.0  # the longest a nightly block may run (all agents)
NIGHTLY_ACTIVE_STALE_H = NIGHTLY_WINDOW_H + 1.0
NIGHTLY_RETRY_H = 1.0  # after a block that was skipped (the QUIET.lock never freed), the next try is this much later


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
    "nightly-real": {"enabled": False, "cmd": _placeholder("nightly-real", "rl/nightly_real.py"), "cwd": "{tree}", "timeout_min": 600,
                     "procs": 12, "every_h": 20.0, "needs": []},
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
        if not due(lab, "nightly", float(cfg.get("every_h", 20.0))):
            return "ran within the last night"
        return nightly_blocked(lab)
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


def store_nightly(readiness: Path, result: dict[str, Any]) -> None:
    """readiness.json `nightly` = the last block's numbers per agent, with the date; every other key stays."""
    doc = read_lenient(readiness)
    doc = doc if isinstance(doc, dict) else {}
    doc["nightly"] = {k: result[k] for k in ("date", "engine", "runs_per_agent", "agents") if k in result}
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
        elif kind == "nightly-real":
            res = read_result(job_dir)
            if res.get("skipped"):  # the QUIET.lock never freed: not a night; try again later
                _atomic(lab / "nightly" / "last-skip", stamp)
            else:
                if res.get("agents"):
                    store_nightly(lab.parent / "readiness.json", res)
                _atomic(lab / "nightly" / "last-ok", stamp)
                (lab / "nightly" / "last-skip").unlink(missing_ok=True)
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
