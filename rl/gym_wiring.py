"""The fight loop's output reaches the run player by itself, or the trainer pauses (D68, Act 3 week, lane 6c).

Until 2026-11-18 a fight brain the gym promoted (`players/vN`) reached the run player only when a worker built the candidate by hand: the
trainer's card hours of twelve days bought nothing. This module is the wiring, three small jobs, all plain files and no emulator:

1. **A promoted player queues its own run candidate** (`queue_candidate`). The champion (`agent/current.json`'s folder) with only its fight
   brain swapped for the player's becomes `agent/gen<k+1>/candidate-fight-v<N>` (`build_fight_candidate`: the champion's folder copied, the
   player's `combat_policy.pt` and `combat_net.pt` in, the ONNX files and model.json made anew, agent.json and train.json written last), and
   the lab's marker `LAB/train/ready-<k>-g<k+1>-fight-v<N>` is written, carrying the lab's keep rule. The lab's train job judges it on fresh
   paired runs; the judge decides, nothing is promoted by the gym. A champion that changes before the next promotion is the next base.
2. **The trainer pauses when nothing uses its output** (`evaluate`). The gym's latest promoted player is *consumed* when its fight network is
   the champion's, or sits in a run candidate that is built and queued (a marker) or judged. A player in neither state 24 hours after its
   promotion writes `LAB/alerts/gym-unconsumed.txt` and the daemon stops generating, gating and training until the player is consumed or the
   orchestrator removes the file (a removed file is an acknowledgement: that player is never alerted on again).
3. **The gate's gain is read against the run judge's verdict** (`join_judged`): when the candidate that carried a player has a
   `promotion.json`, its Act 3 kills and wins, its verdict and its pairs are joined to the gate's promotion row (`gate/run-judge.jsonl`, one
   `run judge:` line in loop.log, and `gym_clock`'s daily line).

Pure bookkeeping on paths it is given (`Wiring`) and an injectable builder, so the tests need no engine. The builder itself runs in the GPU
venv (`python rl/gym_wiring.py build ...`: ONNX export needs torch and onnx), started by the daemon as a short child process.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import os
import re
import shutil
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

REF = Path(os.environ.get("SPIRELINE_REFEREE", "D:/spireline/referee"))
GPU_PYTHON = os.environ.get("SPIRELINE_GPU_PYTHON", "D:/spireline/venv-gpu/Scripts/python.exe")

ALERT_FILE = "gym-unconsumed.txt"
UNCONSUMED_H = 24.0  # a promoted player in no champion and no queued or judged candidate this long after its promotion stops the loop
TAG_FMT = "fight-{version}"  # the candidate's tag: gen<k+1>/candidate-fight-v<N>, marker ready-<k>-g<k+1>-fight-v<N>
SWAP_NOT_COPIED = ("agent.json", "train.json", "promotion.json", "model.json")  # written anew, or the base's own record
FIGHT_FILES = ("combat_policy.pt", "combat_net.pt", "combat_net.onnx")  # screens.FIGHT_FILES: the player's, never the base's
KEPT_ONNX = ("fight_net.onnx", "run_value.onnx")  # the export reproduces the base's bytes; where it does not, the base's stay
BUILD_TIMEOUT_S = 1500.0
_HOME = re.compile(r"(?i)[a-z]:[\\/]+users[\\/]+[^\\/\s'\"]+")
_VERSION = re.compile(r"v\d+")


class WiringError(RuntimeError):
    """The candidate could not be built or queued; the daemon logs it and tries again at the next check."""


def clean(text: str) -> str:
    return _HOME.sub("<home>", str(text))


def utc_text(t: dt.datetime | None = None) -> str:
    return (t or dt.datetime.now(dt.timezone.utc)).strftime("%Y-%m-%dT%H:%M:%SZ")


def parse_stamp(s) -> dt.datetime | None:
    try:
        return dt.datetime.strptime(str(s), "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)
    except ValueError:
        return None


def read_json(p: Path | str, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(p: Path | str, doc) -> None:
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, p)


def file_sha256(p: Path | str) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


@dataclass
class Wiring:
    """Where the lab's files are, and the builder. `builder(base, player_dir, agent_root, target, tag, why) -> candidate folder`."""

    agent_root: Path = REF / "agent"
    train_dir: Path = REF / "LAB" / "train"
    alerts_dir: Path = REF / "LAB" / "alerts"
    jobs_json: Path = REF / "LAB" / "jobs.json"
    partial_dir: Path | None = None  # where a half-built folder is moved (default: beside the train markers' lab, never under agent/)
    unconsumed_h: float = UNCONSUMED_H
    builder: Callable[..., Path] | None = None

    def __post_init__(self) -> None:
        self.agent_root, self.train_dir, self.alerts_dir = Path(self.agent_root), Path(self.train_dir), Path(self.alerts_dir)
        self.jobs_json = Path(self.jobs_json)

    @property
    def alert_path(self) -> Path:
        return self.alerts_dir / ALERT_FILE


# --------------------------------------------------------------------------------------------------- what the files say


_HASHED: dict[tuple[str, int, int], str] = {}  # (path, mtime, size) -> sha256: a folder with no model.json is hashed once, not at every look


def _hashed(p: Path) -> str:
    st = p.stat()
    key = (str(p), st.st_mtime_ns, st.st_size)
    if key not in _HASHED:
        _HASHED[key] = file_sha256(p)
    return _HASHED[key]


def policy_sha(folder: Path | str) -> str | None:
    """The sha256 of a folder's fight policy: model.json's `policy_sha256` (a player's and a candidate's both carry it), else a built
    candidate's agent.json `built.prior.policy_sha256`, else the bytes of combat_policy.pt. None when there is none."""
    d = Path(folder)
    sha = (read_json(d / "model.json") or {}).get("policy_sha256")
    if not sha:
        sha = ((((read_json(d / "agent.json") or {}).get("built") or {}).get("prior")) or {}).get("policy_sha256")
    if not sha and (d / "combat_policy.pt").is_file():
        sha = _hashed(d / "combat_policy.pt")
    return str(sha) if sha else None


def champion(agent_root: Path | str) -> dict | None:
    """The run player now: {"generation", "dir", "sha"} from `current.json`; None when it names no folder that exists."""
    doc = read_json(Path(agent_root) / "current.json")
    if not isinstance(doc, dict) or not doc.get("dir"):
        return None
    d = Path(str(doc["dir"]).replace("\\", "/"))
    if not d.is_dir():
        return None
    return {"generation": int(doc.get("generation", (read_json(d / "agent.json") or {}).get("generation", 0))), "dir": d, "sha": policy_sha(d)}


def candidate_folders(agent_root: Path | str) -> list[Path]:
    return sorted(p for p in Path(agent_root).glob("gen*/candidate*") if p.is_dir())


def carrying(agent_root: Path | str, sha: str | None) -> list[dict]:
    """The candidate folders (`agent/gen*/candidate*`) whose fight policy is `sha`, with whether it was built with that network in place of its
    base's (`introduced`: a candidate that only inherited it from its base, as every later candidate of a champion's line does, is not the
    one that carried the player) and whether the run judge has spoken (`promotion.json`) and what it said."""
    out = []
    if not sha:
        return out
    for d in candidate_folders(agent_root):
        if policy_sha(d) != sha:
            continue
        doc = read_json(d / "promotion.json")
        meta = read_json(d / "agent.json") or {}
        base_sha = ((((meta.get("built") or {}).get("base")) or {}).get("policy_sha256"))
        out.append({"dir": d, "rel": f"{d.parent.name}/{d.name}", "complete": (d / "agent.json").is_file(),
                    "generation": meta.get("generation"), "built_on": meta.get("built_on"),
                    "introduced": bool(base_sha) and base_sha != sha,  # built with this fight network in place of its base's, not inherited from it
                    "judged": isinstance(doc, dict), "doc": doc if isinstance(doc, dict) else None})
    return out


def marker_names(train_dir: Path | str, version: str) -> list[str]:
    """The lab's markers (ready- or done-) of the candidate tagged for this player."""
    pat = re.compile(r"(ready|done)-\d+-g\d+-" + re.escape(TAG_FMT.format(version=version)))
    d = Path(train_dir)
    return sorted(p.name for p in d.iterdir() if pat.fullmatch(p.name)) if d.is_dir() else []


def consumption(w: Wiring, version: str, player_dir: Path | str) -> dict:
    """Who uses the player: "champion" (its fight policy is the champion's), "judged" (a candidate carrying it has a promotion.json),
    "queued" (a ready- or done- marker names it) or "none" (no candidate, or one built and not yet queued)."""
    sha = policy_sha(player_dir)
    out = {"version": version, "sha": sha, "state": "none", "via": None}
    if not sha:
        out["via"] = "the player has no fight policy to compare"
        return out
    champ = champion(w.agent_root)
    if champ and champ["sha"] == sha:
        return {**out, "state": "champion", "via": f"generation {champ['generation']}"}
    cands = carrying(w.agent_root, sha)
    judged = [c for c in cands if c["judged"]]
    if judged:
        return {**out, "state": "judged", "via": judged[-1]["rel"]}
    marks = marker_names(w.train_dir, version)
    if marks:
        return {**out, "state": "queued", "via": marks[-1]}
    if cands:
        out["via"] = f"{cands[-1]['rel']} is built and has no marker"
    return out


def lab_keep_rule(jobs_json: Path | str) -> str | None:
    """The rule the lab's train job judges by now: `train.keep_rule` in jobs.json when it has the field, else the `--keep-rule` of the train
    command; None when neither says (the marker then carries no rule and the lab's own is used)."""
    cfg = read_json(jobs_json)
    train = cfg.get("train") if isinstance(cfg, dict) and isinstance(cfg.get("train"), dict) else {}
    if train.get("keep_rule"):
        return str(train["keep_rule"])
    cmd = [str(x) for x in (train.get("cmd") or [])]
    for i, x in enumerate(cmd):
        if x == "--keep-rule" and i + 1 < len(cmd):
            return cmd[i + 1]
        if x.startswith("--keep-rule="):
            return x.split("=", 1)[1]
    return None


def marker_text(now: dt.datetime | None, keep_rule: str | None) -> str:
    """A marker's text: the UTC stamp on the first line (as every marker has had), then `keep_rule: <rule>` when the lab's rule is known."""
    return utc_text(now) + "\n" + (f"keep_rule: {keep_rule}\n" if keep_rule else "")


# --------------------------------------------------------------------------------------------------- the builder


def build_fight_candidate(base: Path | str, player_dir: Path | str, agent_root: Path | str, target: int, *, why: str, tag: str,
                          export: bool = True) -> Path:
    """Generation `target`'s candidate that is the generation `base` (a generation folder or a current.json) with its fight brain replaced
    by the gym player in `player_dir` (`players/v<N>`: model.json, combat_policy.pt, combat_net.pt) and nothing else: every other file of
    the base's folder is copied byte for byte (V_run fit, human prior, run student and its legs, the ONNX of the run value), agent.json keeps
    the base's macro, recipe, blend, student, stakes schedule, hold and mask version, so the candidate differs in the fight part alone
    (`screens.differing` reads ["fight"]: the cheap fight screen applies). model.json is written for `target` (`write_prior_model_json`),
    combat_net.onnx is exported from the player (`export_all`; without `export` the candidate has none), and the run value's ONNX stay the
    base's bytes. agent.json is written last, so a folder with none is a half-built one. Written as agent/gen<target>/candidate-<tag>
    (`gen_loop.candidate_dirname`), never over an existing folder; a failure removes what this call wrote."""
    sys.path[:0] = [p for p in (str(ROOT / "rl"), str(ROOT / "src"), str(ROOT)) if p not in sys.path]
    import combat_prior  # noqa: PLC0415
    import gen_loop  # noqa: PLC0415
    import screens  # noqa: PLC0415

    player_dir, agent_root, target = Path(player_dir), Path(agent_root), int(target)
    base_agent = gen_loop.load_agent(base)
    if base_agent.path is None:
        raise ValueError("the base must be a generation directory or a current.json, not generation 0's empty agent")
    if target <= base_agent.generation:
        raise ValueError(f"a candidate built on generation {base_agent.generation} cannot be generation {target}")
    base_dir = Path(base_agent.path)
    champ = combat_prior.find(player_dir)  # PriorError when this build cannot play it
    if champ is None:
        raise FileNotFoundError(f"no combat prior (model.json and combat_policy.pt) in {player_dir.name}")
    if champ.probe or {"c_puct": champ.c_puct, "prior_min_visits": champ.prior_min_visits} != gen_loop.PRIOR_SEARCH:
        raise ValueError(f"{player_dir.name} does not search with the settings every candidate's model.json carries")
    for name in ("combat_policy.pt", "combat_net.pt"):
        if not (player_dir / name).is_file():
            raise FileNotFoundError(f"{player_dir.name} has no {name}")
    cand = agent_root / f"gen{target}" / gen_loop.candidate_dirname(0, 1, tag)
    if cand.exists() and any(cand.iterdir()):
        raise FileExistsError(f"gen{target}/{cand.name} exists; a built candidate never writes over one")
    base_sha = base_agent.describe()["prior_sha256"]
    cand.mkdir(parents=True, exist_ok=True)
    try:
        for f in sorted(base_dir.iterdir()):
            if f.is_file() and f.name not in SWAP_NOT_COPIED and f.name not in FIGHT_FILES:
                shutil.copyfile(f, cand / f.name)
        for name in ("combat_policy.pt", "combat_net.pt"):
            shutil.copyfile(player_dir / name, cand / name)
        model = gen_loop.write_prior_model_json(cand, target)
        v_sha = gen_loop.file_sha256(cand / "combat_policy.pt")
        if not (model["policy_sha256"] == v_sha == champ.policy_sha256):
            raise ValueError(f"the exported model.json does not name the player's policy ({model['policy_sha256'][:12]} against {v_sha[:12]})")
        onnx = None
        if export:
            onnx = gen_loop.export_all(cand)  # combat_net.onnx from the player; the run value's two from run_value.pt (the base's bytes)
            for n in KEPT_ONNX:
                if (base_dir / n).is_file() and gen_loop.file_sha256(cand / n) != gen_loop.file_sha256(base_dir / n):
                    shutil.copyfile(base_dir / n, cand / n)
                    onnx[n.split(".")[0]] = {**(onnx.get(n.split(".")[0]) or {}), "kept": "the base's file"}
        net_sha = gen_loop.file_sha256(cand / "combat_net.pt")
        base_meta = gen_loop.read_json(base_dir / "agent.json") if (base_dir / "agent.json").is_file() else {}
        player_train = gen_loop.read_json(player_dir / "train.json") if (player_dir / "train.json").is_file() else None
        player_model = gen_loop.read_json(player_dir / "model.json")
        label = (player_model.get("gym_loop") or {}).get("checkpoint")
        prior = {"dir": player_dir.as_posix(), "policy_sha256": v_sha, "net_sha256": net_sha}
        run_value = cand / "run_value.npz"
        base_block = {"dir": base_dir.as_posix(), "generation": base_agent.generation, "macro": base_agent.macro, "policy_sha256": base_sha,
                      "run_value_sha256": gen_loop.file_sha256(run_value) if run_value.is_file() else None}
        lineage = {"from_generation": base_agent.generation, "skips": list(range(base_agent.generation + 1, target)),
                   "difference": (f"generation {base_agent.generation} with the fight loop's player {player_dir.name}"
                                  + (f" (checkpoint {label})" if label else "") + f" in place of its fight network ({base_sha[:12]}); nothing else changed"),
                   "decision": why}
        note = (f"built, not trained: generation {base_agent.generation}'s agent ({base_agent.recipe() or base_agent.macro}) with the fight loop's "
                f"player {player_dir.name} (policy {v_sha[:12]}) in place of its fight network ({base_sha[:12]}) and nothing else; its E-block "
                "judges the fight network alone in whole runs" + ("; the V_run fit was made on runs played by the old fight network"
                                                                 if (base_meta.get("vrun_fit") or {}).get("prior_sha256") not in (None, v_sha) else ""))
        meta = {k: v for k, v in base_meta.items() if k not in ("lineage", "note", gen_loop.BUILT_KEY, "vrun_fit")}
        if isinstance(base_meta.get("vrun_fit"), dict):
            fit = dict(base_meta["vrun_fit"])
            fit["matched"] = fit.get("prior_sha256") == v_sha
            meta["vrun_fit"] = fit
        meta.update(generation=target, built_on=base_agent.generation, written=gen_loop.utc_now(), lineage=lineage)
        meta[gen_loop.BUILT_KEY] = {"why": why, "trained": False, "swap": "fight", "prior": prior, "base": base_block, "tag": tag}
        meta["note"] = note
        built = {"why": why, "written": meta["written"], "trained": False, "swap": "fight",
                 "rule": "the base's folder copied unchanged but its fight network (combat_policy.pt, combat_net.pt, combat_net.onnx, model.json); "
                         "`gen0.py train` judges the candidate as it is",
                 "prior": {**prior, "model_json": player_model}, "base": base_block, "tag": tag}
        doc = {"candidate_generation": target, gen_loop.BUILT_KEY: built, "lineage": lineage, "prior_train": player_train,
               "base_train": gen_loop.read_json(base_dir / "train.json") if (base_dir / "train.json").is_file() else None,
               "model_json": model, **({"onnx": onnx} if onnx is not None else {})}
        gen_loop.write_json(cand / "train.json", doc)
        gen_loop.write_json(cand / "agent.json", meta)  # last: `built_candidate` reads a complete directory
        got = gen_loop.load_agent(cand)
        if (got.generation, got.macro, got.recipe(), got.describe()["prior_sha256"]) != (target, base_agent.macro, base_agent.recipe(), v_sha):
            raise ValueError(f"gen{target}/{cand.name} does not load as the base with the player's fight network: {got.describe()}")
        if screens.differing(cand, base_dir) != ["fight"]:
            raise ValueError(f"gen{target}/{cand.name} differs from the base in {screens.differing(cand, base_dir)}, not the fight network alone")
    except BaseException:
        shutil.rmtree(cand, ignore_errors=True)
        raise
    return cand


def subprocess_builder(base: Path, player_dir: Path, agent_root: Path, target: int, tag: str, why: str) -> Path:
    """Run `build` in the GPU venv (ONNX export needs torch and onnx; the daemon's own venv has no onnx). The child is in the daemon's Job
    Object, so it ends with it."""
    argv = [GPU_PYTHON, str(ROOT / "rl" / "gym_wiring.py"), "build", "--base", str(base), "--player-dir", str(player_dir),
            "--agent-root", str(agent_root), "--target", str(int(target)), "--tag", tag, "--why", why]
    env = dict(os.environ)
    if (ROOT / "out").is_dir():
        env.setdefault("STS2_LIB_PATH", str(ROOT / "out"))
    env.setdefault("STS2_ALLOW_STALE_NATIVE", "1")
    r = subprocess.run(argv, cwd=str(ROOT), env=env, capture_output=True, text=True, encoding="utf-8", errors="replace", check=False,
                       timeout=BUILD_TIMEOUT_S, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        raise WiringError(clean(f"the builder exited {r.returncode}: {(r.stderr.strip() or r.stdout.strip())[-400:]}"))
    last = [x for x in r.stdout.splitlines() if x.strip().startswith("{")]
    try:
        return Path(json.loads(last[-1])["candidate_dir"])
    except (IndexError, KeyError, ValueError) as error:
        raise WiringError(f"the builder printed no candidate: {clean(r.stdout.strip()[-300:])}") from error


# --------------------------------------------------------------------------------------------------- task 1: queue the candidate


def _move_aside(w: Wiring, folder: Path, now: dt.datetime) -> Path:
    base = Path(w.partial_dir) if w.partial_dir else Path(w.train_dir).parent.parent / "gym-loop" / "partial-builds"
    base.mkdir(parents=True, exist_ok=True)
    dest = base / f"{folder.parent.name}-{folder.name}-{now.strftime('%Y%m%dT%H%M%SZ')}"
    shutil.move(str(folder), str(dest))
    return dest


def queue_candidate(w: Wiring, version: str, player_dir: Path | str, *, checkpoint: str | None = None, promoted: str | None = None,
                    now: dt.datetime | None = None, log: Callable[[str], None] = lambda _m: None) -> dict:
    """Build `agent/gen<k+1>/candidate-fight-<version>` on the champion (k its generation) and write the lab's marker
    `ready-<k>-g<k+1>-fight-<version>` with the lab's keep rule. Idempotent: a built folder that matches is not built again, and a marker
    that exists (or was consumed) is never written again. A champion that already has the player's fight network needs neither. A half-built
    folder, or one built on a champion that has since changed and never queued, is moved aside and built again. Returns what was done:
    {"candidate", "marker", "keep_rule", "built"}. Raises `WiringError` when it cannot."""
    now = now or dt.datetime.now(dt.timezone.utc)
    player_dir = Path(player_dir)
    sha = policy_sha(player_dir)
    if not sha or not (player_dir / "combat_policy.pt").is_file():
        raise WiringError(f"{player_dir.name} has no fight policy to build on")
    champ = champion(w.agent_root)
    if champ is None:
        raise WiringError("agent/current.json names no champion folder")
    if champ["sha"] == sha:
        return {"candidate": None, "marker": None, "keep_rule": None, "built": False, "note": "the champion already has this fight network"}
    k = champ["generation"]
    target, tag = k + 1, TAG_FMT.format(version=version)
    cand = w.agent_root / f"gen{target}" / f"candidate-{tag}"
    marker = w.train_dir / f"ready-{k}-g{target}-{tag}"
    marks = marker_names(w.train_dir, version)
    built = False
    if cand.exists() and any(cand.iterdir()):
        meta = read_json(cand / "agent.json")
        usable = isinstance(meta, dict) and meta.get("built_on") == k and policy_sha(cand) == sha
        if not usable and not marks and not (cand / "promotion.json").is_file():
            dest = _move_aside(w, cand, now)
            log(f"wiring: {version}: {cand.parent.name}/{cand.name} was not a complete build on generation {k} ({'half-built' if not meta else 'built on ' + str(meta.get('built_on'))}); moved aside to {dest.name}")
        elif not usable:
            return {"candidate": cand.as_posix(), "marker": marks[-1] if marks else None, "keep_rule": None, "built": False,
                    "note": "a candidate folder of this name stands (queued or judged earlier)"}
    if not cand.exists() or not any(cand.iterdir()):
        why = (f"the fight loop's player {version}" + (f" (checkpoint {checkpoint})" if checkpoint else "") + (f", promoted {promoted}" if promoted else "")
               + ": queued by the gym loop for a whole-run judgement (D68, lane 6c)")
        build = w.builder or subprocess_builder
        try:
            got = Path(build(champ["dir"], player_dir, w.agent_root, target, tag, why))
        except WiringError:
            raise
        except Exception as error:  # noqa: BLE001 -- a builder's failure is the wiring's to log, not the daemon's to die of
            raise WiringError(clean(f"{type(error).__name__}: {str(error)[-300:]}")) from error
        if got.resolve() != cand.resolve() or not (cand / "agent.json").is_file():
            raise WiringError(f"the builder left {got} where {cand} was wanted, or no agent.json")
        built = True
    if marks:
        return {"candidate": cand.as_posix(), "marker": marks[-1], "keep_rule": None, "built": built, "note": "a marker for this player exists"}
    rule = lab_keep_rule(w.jobs_json)
    marker.parent.mkdir(parents=True, exist_ok=True)
    tmp = marker.with_name("." + marker.name + ".tmp")  # not matched by the lab's `ready-*` glob until it is whole
    tmp.write_text(marker_text(now, rule), encoding="utf-8")
    os.replace(tmp, marker)
    log(f"wiring: {version} queued as {cand.parent.name}/{cand.name}, marker {marker.name}" + (f" (keep rule {rule})" if rule else " (the lab's keep rule not known: no rule on the marker)"))
    return {"candidate": cand.as_posix(), "marker": marker.name, "keep_rule": rule, "built": built}


# --------------------------------------------------------------------------------------------------- task 3: the run judge's verdict


def run_judge_summary(doc: dict) -> dict:
    """What a candidate's promotion.json says, for the gate's books: the verdict, the pairs, the kills of the Act 1, Act 2 and first Act 3
    boss (candidate and base) and the wins (candidate and base). Fields a document lacks are None."""
    keep = doc.get("keep_v2") if isinstance(doc.get("keep_v2"), dict) else {}
    kills = keep.get("kills") if isinstance(keep.get("kills"), dict) else {}
    wins = doc.get("wins") if isinstance(doc.get("wins"), dict) else {}
    dec = doc.get("decision") if isinstance(doc.get("decision"), dict) else {}

    def pair(v):
        return [int(v[0]), int(v[1])] if isinstance(v, (list, tuple)) and len(v) == 2 else None

    return {"promoted": dec.get("promote") if isinstance(dec.get("promote"), bool) else None, "pairs": doc.get("paired_seeds"),
            "keep_rule": doc.get("keep_rule"), "rule": dec.get("rule"), "written": doc.get("written"), "seed_block": doc.get("seed_block"),
            "act1_boss_kills": pair(kills.get("B1")), "act2_boss_kills": pair(kills.get("B2")), "act3_boss_kills": pair(kills.get("B3a")),
            "wins": pair(kills.get("win")) or ([wins.get("cand"), wins.get("base")] if "cand" in wins and "base" in wins else None)}


def judge_line(version: str, rel: str, s: dict, checkpoint: str | None = None) -> str:
    def pr(p, what):
        return f"{what} {p[0]} against {p[1]}" if p else f"{what} unknown"

    verdict = {True: "promoted", False: "not promoted", None: "no verdict read"}[s["promoted"]]
    return (f"run judge: {version}" + (f" (gate {checkpoint})" if checkpoint else "") + f" carried by {rel}: {verdict}"
            + (f" under keep rule {s['keep_rule']}" if s.get("keep_rule") else "") + f" on {s.get('pairs') or '?'} pairs; "
            + pr(s["act3_boss_kills"], "first Act 3 boss kills") + ", " + pr(s["wins"], "wins") + " (candidate against base)")


def join_judged(w: Wiring, ws: dict, version: str, player_dir: Path | str, *, checkpoint: str | None, gates_dir: Path | None,
                now: dt.datetime, log: Callable[[str], None]) -> list[dict]:
    """The run judge's verdict on every candidate that carried this player and has a promotion.json, once per (candidate, judgement time):
    a line in loop.log, a row in `gate/run-judge.jsonl` (the gate's promotion row joined to it by `checkpoint` and `version`), and
    `state["wiring"]["judged"]`. Returns the rows written now."""
    done = ws.setdefault("judged", {})
    out = []
    for c in carrying(w.agent_root, policy_sha(player_dir)):
        if not c["judged"] or not c["introduced"]:
            continue
        s = run_judge_summary(c["doc"])
        key = f"{c['rel']}@{s['written']}"
        if key in done.get(version, []):
            continue
        row = {"t": utc_text(now), "version": version, "checkpoint": checkpoint, "candidate": c["rel"], **s}
        if gates_dir is not None:
            with open(Path(gates_dir) / "run-judge.jsonl", "a", encoding="utf-8") as fh:
                fh.write(json.dumps(row) + "\n")
        done.setdefault(version, []).append(key)
        log(judge_line(version, c["rel"], s, checkpoint))
        out.append(row)
    return out


# --------------------------------------------------------------------------------------------------- task 2: the pause


def alert_text(version: str, promoted: str, age_h: float, cons: dict, now: dt.datetime, hours: float, error: str | None) -> str:
    return (f"ALERT gym-unconsumed: the fight loop's player {version} (promoted {promoted}, {age_h:.1f} hours ago) is in neither the champion's fight "
            f"network nor a queued or judged run candidate; raised {utc_text(now)}\n"
            f"The loop (generation, gate, trainer) is stopped: nothing may spend the card on a fight brain that no run uses (D68, lane 6c). "
            f"It resumes by itself when the player is consumed (the champion's fight network is the player's, or a candidate carrying it has a "
            f"marker or a promotion.json), or when this file is removed (the player is then not alerted on again). "
            f"Limit: {hours:g} hours from the promotion.\n"
            + (f"Why the candidate was not queued: {error}\n" if error else "") + f"State: {cons['state']} ({cons.get('via')}).\n"
            f"By hand: python rl/gym_wiring.py queue --root <the loop's folder> after fixing the cause.\n")


def evaluate(w: Wiring, state: dict, now: dt.datetime, *, log: Callable[[str], None], gates_dir: Path | None = None,
             queue: bool = True) -> dict:
    """One look at the latest promoted player (`state["promotions"][-1]`), changing `state["wiring"]` (the caller saves the state):

    - the player is consumed: nothing to do (any alert of ours is removed);
    - not consumed and `queue`: the candidate is built and its marker written (`queue_candidate`); a failure is logged once per kind of failure;
    - still not consumed `unconsumed_h` after its promotion: the alert is written and `paused` is True; an alert file the orchestrator removed
      is an acknowledgement (the player is never alerted on again, the loop resumes);
    - every judged candidate that carried a promoted player is joined to the gate's books (`join_judged`).

    Returns {"version", "consumption", "paused" (reason or None), "queued" (what `queue_candidate` returned), "error"}."""
    ws = state.setdefault("wiring", {})
    out: dict = {"version": None, "consumption": None, "paused": None, "queued": None, "error": None}
    promos = state.get("promotions") or []
    for p in promos:  # the run judge's word on every player the gym promoted (v1 and v2 were never carried: they find nothing)
        ver = p.get("version")
        if ver in state.get("versions", {}):
            pdir = Path(state["versions"][ver]["dir"])
            if (pdir / "model.json").is_file():
                join_judged(w, ws, ver, pdir, checkpoint=p.get("checkpoint"), gates_dir=gates_dir, now=now, log=log)
    if not promos:
        return out
    last = promos[-1]
    ver = last["version"]
    out["version"] = ver
    if ver not in state.get("versions", {}) or not (Path(state["versions"][ver]["dir"]) / "model.json").is_file():
        out["error"] = f"{ver} has no folder to read"
        return out
    pdir = Path(state["versions"][ver]["dir"])
    cons = consumption(w, ver, pdir)
    if cons["state"] == "none" and queue:
        try:
            out["queued"] = queue_candidate(w, ver, pdir, checkpoint=last.get("checkpoint"), promoted=last.get("t"), now=now, log=log)
            ws.pop("build_error", None)
        except WiringError as error:
            out["error"] = str(error)
            if ws.get("build_error") != out["error"]:  # said once, not at every check
                ws["build_error"] = out["error"]
                log(f"wiring: {ver}: not queued: {out['error']}")
        cons = consumption(w, ver, pdir)
    out["consumption"] = cons
    if cons["state"] != "none":
        if w.alert_path.is_file():
            w.alert_path.unlink()
            log(f"wiring: {ver} is consumed ({cons['state']}: {cons.get('via')}); removed {w.alert_path.parent.name}/{w.alert_path.name}; the loop resumes")
        ws.pop("alert", None)
        ws.pop("build_error", None)
        return out
    acked = ws.setdefault("acked", [])
    alert = ws.get("alert")
    if alert and alert.get("version") == ver and not w.alert_path.is_file():  # the orchestrator removed it
        acked.append(ver)
        ws.pop("alert", None)
        log(f"wiring: the alert for {ver} was removed; the loop resumes and {ver} is not alerted on again")
    if ver in acked:
        return out
    promoted_at = parse_stamp(last.get("t"))
    age_h = (now - promoted_at).total_seconds() / 3600.0 if promoted_at else 0.0
    if promoted_at is not None and age_h >= float(w.unconsumed_h):
        existed = w.alert_path.is_file()
        w.alert_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = w.alert_path.with_name(w.alert_path.name + ".tmp")
        tmp.write_text(alert_text(ver, str(last.get("t")), age_h, cons, now, float(w.unconsumed_h), out["error"] or ws.get("build_error")),
                       encoding="utf-8", newline="\n")
        os.replace(tmp, w.alert_path)
        ws["alert"] = {"version": ver, "t": (alert or {}).get("t") or utc_text(now)}
        if not existed:
            log(f"wiring: {ver} unconsumed {age_h:.1f} h after its promotion ({cons['state']}: {cons.get('via')}); wrote "
                f"{w.alert_path.parent.name}/{w.alert_path.name}; generation, gate and trainer stop until it is consumed or the file is removed")
        out["paused"] = f"{ver} unconsumed {age_h:.0f} h after its promotion"
    return out


# --------------------------------------------------------------------------------------------------- command line


def _cmd_build(a: argparse.Namespace) -> int:
    lib = os.environ.get("STS2_LIB_PATH")
    if not lib:
        lib = next((str(d) for d in (ROOT / "out", ROOT.parent / "sts2" / "out") if (d / "Sts2Emulator.dll").exists()), "")
        if lib:
            os.environ["STS2_LIB_PATH"] = lib
    os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")
    cand = build_fight_candidate(a.base, a.player_dir, a.agent_root, a.target, why=a.why, tag=a.tag, export=not a.no_export)
    print(json.dumps({"candidate_dir": Path(cand).as_posix(), "candidate": f"{cand.parent.name}/{cand.name}"}))
    return 0


def _cmd_queue(a: argparse.Namespace) -> int:
    """By hand: queue the latest promoted player of a loop folder (the same `queue_candidate` the daemon calls) and print whether it is consumed.
    Writes nothing of the loop's own (state.json belongs to the daemon): the state of a player is read from the files."""
    state = read_json(Path(a.root) / "state.json") or {}
    promos = state.get("promotions") or []
    if not promos:
        print(json.dumps({"version": None, "note": "the gym has promoted nothing"}))
        return 0
    last = promos[-1]
    ver = last["version"]
    pdir = Path(state["versions"][ver]["dir"])
    w = Wiring(agent_root=a.agent_root, train_dir=a.train_dir, alerts_dir=a.alerts_dir, jobs_json=a.jobs_json)
    before = consumption(w, ver, pdir)
    done = None
    if before["state"] == "none" and not a.dry_run:
        done = queue_candidate(w, ver, pdir, checkpoint=last.get("checkpoint"), promoted=last.get("t"), log=print)
    print(json.dumps({"version": ver, "consumption": before, "queued": done, "consumption_now": consumption(w, ver, pdir)}, default=str))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build", help="the champion's folder with the gym player's fight network, as agent/gen<target>/candidate-<tag>")
    b.add_argument("--base", required=True, help="a generation folder or a current.json")
    b.add_argument("--player-dir", required=True, help="the gym's players/v<N>")
    b.add_argument("--agent-root", required=True)
    b.add_argument("--target", type=int, required=True)
    b.add_argument("--tag", required=True)
    b.add_argument("--why", required=True)
    b.add_argument("--no-export", action="store_true")
    q = sub.add_parser("queue", help="by hand: queue the latest promoted player of a loop folder if it is not consumed, and report")
    q.add_argument("--root", default=str(REF / "gym-loop"))
    q.add_argument("--agent-root", default=str(REF / "agent"))
    q.add_argument("--train-dir", default=str(REF / "LAB" / "train"))
    q.add_argument("--alerts-dir", default=str(REF / "LAB" / "alerts"))
    q.add_argument("--jobs-json", default=str(REF / "LAB" / "jobs.json"))
    q.add_argument("--dry-run", action="store_true", help="read and report; build nothing and write no marker")
    return ap


def main(argv=None) -> int:
    a = build_parser().parse_args(argv)
    return {"build": _cmd_build, "queue": _cmd_queue}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
