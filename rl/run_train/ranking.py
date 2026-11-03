"""The ranking check of the V_run keep test (D47, D49): how a candidate head ranks the options in front of it, against the head it
would replace.

The D33 keep test grades outcome forecasts (the out-of-fold Act 1 boss-kill Brier skill). Generation 12's fit passed it and was then
refused by 5.9 floors: it rested a third less and healed half as often before the first boss, and the forecast test could not see
that. This check ranks the options of one fixed set of decisions with the candidate and with the incumbent and compares them:

  * the set: decisions the lab's self-play shards hold (rest sites, card rewards, map choices), chosen by a fixed seed from a fixed
    pool (the generation-9 shards: a finished set no later fit is made from), written beside the keep document so any later run
    reads the same decisions (`ranking_set.json`; `set_digest` names it). Runs whose seed is in the candidate's fit window are left
    out, so the decisions are held out of the fit;
  * the measures, on the same states for both heads: the share of decisions where the candidate's top choice is the incumbent's; the
    rest rate (map choices that offer a rest site beside another room: the share where the head goes to the rest site); the heal
    rate (rest sites: the share where the head heals instead of upgrading); the card skip rate; the elite rate; and, from the human
    leg, the share of the held-out human fifth's card-reward and rest-site decisions where the head's choice is the human's;
  * the rule (`judge`): the candidate fails when its rest rate or heal rate falls more than `MARGIN_POINTS` below the incumbent's,
    or when its human agreement falls below the incumbent's by more than `HUMAN_Z` standard errors of the paired difference.

The maths takes plain arrays and lists; `build_set` and `load_set` read the shards, `rank` runs a head over the set on the CPU.
Human data enters only as the aggregate counts of the human leg (`HUMAN_TOOL`), never a single run.
"""

from __future__ import annotations

import dataclasses
import gzip
import hashlib
import json
import math
import os
import random
import subprocess
import sys
import time
from collections.abc import Callable, Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

SET_FILE = "ranking_set.json"  # the fixed set, written beside the keep document
RECORD_FILE = "ranking.json"  # the check's whole record, beside the keep document too
SET_SEED = 20261103  # chosen once (2026-11-03); changing it changes the set, so it changes the margins' meaning
POOL_GENERATIONS = (9,)  # the self-play shards the set is drawn from: finished, played by the generation the incumbent heads were made for
QUOTAS = {"rest": 3000, "card_reward": 3000, "map_rest": 3000, "map_other": 2000}  # decisions per stratum
PER_RUN = {"rest": 2, "card_reward": 2, "map_rest": 2, "map_other": 1}  # at most this many per stratum from one run
W_WON = 53.0  # the lookahead macro's worth of a won run (the policy block of every record; policy.Model needs it)
NODE_ELITE, NODE_REST = 2, 3  # run_train.vocab.NODE_*

# The rule. Margins in points of a rate (0.05 = 5 points) on the same states, chosen on the V_run heads on disk against generation 9's
# (2026-11-03, this set, the pool's 11,000 decisions; the HANDOFF line of D49 has the table). The moves of the other heads on disk
# (generations 5 to 10, 12) against generation 9's: heal rate -8.3 (generation 10's refit, refused) to +5.3 points, rest rate -3.0 to
# +6.2; generation 9 against generation 8, both promoted: heal -5.3, rest -2.4. The margins sit just beyond those: 10 points of heal
# rate, 5 of rest rate.
# Generation 12's refused fit: heal rate -26.4 points (0.200 against 0.464), rest rate -3.4 (0.831 against 0.865), human agreement
# -2.9 points at card rewards (z -6.0) and -4.9 at rest sites (z -4.7); generation 11's head is generation 9's own file (all zero).
MARGIN_POINTS = {"rest_rate": 0.05, "heal_rate": 0.10}
HUMAN_Z = 2.0  # human agreement may fall this many standard errors of the paired difference below the incumbent's
HUMAN_MIN_DECISIONS = 1000  # fewer decisions on a human measure and it cannot fail (it is reported)
RULE_TEXT = ("fails when the candidate's rest rate or heal rate on the set falls more than the margin below the incumbent's, or its "
             "agreement with the human choice on the held-out fifth falls more than HUMAN_Z standard errors below the incumbent's "
             "(paired); passes otherwise (D47, D49)")

FORK_ROOT = Path(__file__).resolve().parents[2]  # the tree this module lives in: its rl/run_train scores the heads in the human leg
HUMAN_TOOL = Path(os.environ.get("SPIRELINE_HUMAN_EXAM_TOOL", "D:/spireline/spireline/tools/boss_entry_exam.py"))
HUMAN_CORPUS = Path(os.environ.get("SPIRELINE_HUMAN_CORPUS", "D:/spireline/data/spire-codex/runs-v0.107.1.jsonl.gz"))
HUMAN_FIFTH_IDS = Path(os.environ.get("SPIRELINE_HUMAN_FIFTH_IDS", "D:/spireline/data/spire-codex/human-exam-fifth-ids.txt"))
HUMAN_MAX_RUNS = 1000  # exam-fifth runs the human leg reads: a fixed seeded third of the fifth (2,924 runs), 14,013 decisions, about 3 min a head

# ------------------------------------------------------------------------------------------------ the set


@dataclasses.dataclass(frozen=True)
class Item:
    """One decision of the set: where it lives and what it offered."""

    stratum: str  # rest card_reward map_rest map_other
    ref: tuple  # (job, shard, record index, decision index)
    run_seed: str
    decision: Any  # run_train.features.Decision
    view: Any  # run_train.features.RunView
    act: int


def _run_train():
    """The fork's run-level package, loaded on first use (numpy only; the native library loads with the package)."""
    from run_train import features, policy, shards  # noqa: PLC0415

    return features, policy, shards


def pool_shards(root: Path | str, generations: Iterable[int] = POOL_GENERATIONS) -> list[tuple[str, Path]]:
    """The run shards of every self-play job under `root` whose manifest names one of `generations`, as (job, path), sorted."""
    want = {int(g) for g in generations}
    out: list[tuple[str, Path]] = []
    for meta in sorted(Path(root).glob("*/shards/shards.json")):
        try:
            man = json.loads(meta.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        if int(man.get("generation", -1)) not in want:
            continue
        names = man.get("run_shards") or sorted(p.name for p in meta.parent.glob("r*.jsonl.gz"))
        out.extend((meta.parent.parent.name, meta.parent / n) for n in sorted(names) if (meta.parent / n).is_file())
    return sorted(out, key=lambda x: (x[0], x[1].name))


def stratum_of(entry: dict[str, Any]) -> str | None:
    """Which stratum a recorded decision belongs to, or None when it is not part of the check (forced, or an unusual screen)."""
    kinds = [o["kind"] for o in entry.get("options", [])]
    t = entry.get("type")
    if t == "rest":
        return "rest" if "rest.heal" in kinds and "rest.smith" in kinds and set(kinds) <= {"rest.heal", "rest.smith"} else None
    if t == "card_reward":
        return "card_reward" if len(kinds) >= 2 and "card.skip" in kinds and set(kinds) <= {"card.take", "card.skip"} else None
    if t == "map" and len(kinds) >= 2 and set(kinds) == {"map.node"}:
        nodes = [o["node"][2] for o in entry["options"]]
        return "map_rest" if NODE_REST in nodes and any(n != NODE_REST for n in nodes) else "map_other"
    return None


def _read_records(path: Path) -> list[dict]:
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def set_digest(doc: dict[str, Any]) -> str:
    """sha256 of the set's identity: the seed, pool, quotas and every reference with its run seed (the order is part of it)."""
    core = {k: doc[k] for k in ("seed", "generations", "quotas", "per_run", "refs")}
    return hashlib.sha256(json.dumps(core, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def build_set(root: Path | str, *, seed: int = SET_SEED, generations: Sequence[int] = POOL_GENERATIONS,
              quotas: dict[str, int] | None = None, per_run: dict[str, int] | None = None,
              exclude_seeds: Iterable[str] = ()) -> tuple[dict[str, Any], list[Item]]:
    """The fixed set of decisions: the manifest (written beside the keep document) and the decisions themselves.

    The pool's shards are put in a seeded order; their records are read in that order and at most `per_run` decisions of each stratum
    are drawn (seeded) from each run, until every stratum holds its quota or the pool is used up. The same root, seed and
    exclusions give the same set. Runs whose seed is in `exclude_seeds` (the candidate's fit window) are skipped."""
    features, _policy, shards = _run_train()
    quotas = dict(QUOTAS if quotas is None else quotas)
    per_run = dict(PER_RUN if per_run is None else per_run)
    skip = {str(s) for s in exclude_seeds}
    files = pool_shards(root, generations)
    rnd = random.Random(seed)
    rnd.shuffle(files)
    items: dict[str, list[Item]] = {k: [] for k in quotas}
    excluded = runs_read = 0
    for job, path in files:
        if all(len(items[k]) >= quotas[k] for k in quotas):
            break
        for ri, rec in enumerate(_read_records(path)):
            if str(rec.get("seed")) in skip:
                excluded += 1
                continue
            runs_read += 1
            found: dict[str, list[int]] = {}
            for di, entry in enumerate(rec["decisions"]):
                st = stratum_of(entry)
                if st in quotas and len(items[st]) < quotas[st]:
                    found.setdefault(st, []).append(di)
            for st in sorted(found):
                take = sorted(rnd.sample(found[st], min(per_run[st], len(found[st]), quotas[st] - len(items[st]))))
                for di in take:
                    entry = rec["decisions"][di]
                    view = shards.state_view(rec, entry)
                    items[st].append(Item(st, (job, path.name, ri, di), str(rec["seed"]), shards.entry_decision(entry), view,
                                          int(entry["state"].get("act_index", 0))))
    refs = [[st, *it.ref, it.run_seed] for st in sorted(items) for it in items[st]]
    doc = {"kind": "ranking-set", "seed": seed, "generations": [int(g) for g in generations], "quotas": quotas, "per_run": per_run,
           "counts": {k: len(v) for k, v in items.items()}, "pool_shards": len(files), "runs_read": runs_read,
           "runs_excluded_in_fit_window": excluded, "refs": refs}
    doc["digest"] = set_digest(doc)
    return doc, [it for k in sorted(items) for it in items[k]]


def set_current(doc: dict[str, Any], exclude_seeds: Iterable[str] = ()) -> bool:
    """Whether a written set is the one this build draws: the same seed, pool and quotas, and none of its runs in the fit window."""
    skip = {str(s) for s in exclude_seeds}
    return (doc.get("seed") == SET_SEED and doc.get("generations") == [int(g) for g in POOL_GENERATIONS] and doc.get("quotas") == QUOTAS
            and doc.get("per_run") == PER_RUN and not any(r[-1] in skip for r in doc.get("refs", ())))


def load_set(doc: dict[str, Any], root: Path | str) -> list[Item]:
    """The decisions a written set names, read back from the shards (raises when a shard is gone or a run's seed has changed)."""
    features, _policy, shards = _run_train()
    if set_digest(doc) != doc.get("digest"):
        raise ValueError("the ranking set's digest does not match its references: the file was changed")
    by_shard: dict[tuple[str, str], list[list]] = {}
    for r in doc["refs"]:
        by_shard.setdefault((r[1], r[2]), []).append(r)
    out: dict[tuple, Item] = {}
    for (job, name), refs in by_shard.items():
        records = _read_records(Path(root) / job / "shards" / name)
        for st, _job, _name, ri, di, run_seed in refs:
            rec = records[ri]
            if str(rec["seed"]) != run_seed:
                raise ValueError(f"{job}/{name} record {ri} is not run {run_seed}: the shard changed since the set was made")
            entry = rec["decisions"][di]
            out[(st, job, name, ri, di)] = Item(st, (job, name, ri, di), run_seed, shards.entry_decision(entry),
                                                shards.state_view(rec, entry), int(entry["state"].get("act_index", 0)))
    return [out[(r[0], r[1], r[2], r[3], r[4])] for r in doc["refs"]]


# ------------------------------------------------------------------------------------------------ ranking


def load_head(npz: Path | str, w: float = W_WON):
    """A V_run head from its run_value.npz, valued in expected floors with a won run worth `w`."""
    _features, policy, _shards = _run_train()
    return dataclasses.replace(policy.load(Path(npz)), W=float(w))


def rank(items: Sequence[Item], model: Any) -> list[list[float | None]]:
    """The head's value of every option of every decision, in the screen's option order (None: unscorable)."""
    _features, policy, _shards = _run_train()
    out: list[list[float | None]] = []
    for it in items:
        scored = {id(s.option): s.value for s in policy.score_options(it.decision, it.view, model)}
        out.append([scored[id(o)] for o in it.decision.options])
    return out


def top_index(values: Sequence[float | None]) -> int:
    """The first best option; unscorable options rank last, and a screen with none scorable returns 0."""
    best, at = -math.inf, 0
    for i, v in enumerate(values):
        if v is not None and v > best:
            best, at = v, i
    return at


def paired(a: Sequence[float], b: Sequence[float]) -> dict[str, float | int | None]:
    """Candidate rate `a`, incumbent rate `b` (0/1 indicators on the same decisions) and the paired difference with its standard error."""
    x, y = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    n = int(x.size)
    if n == 0:
        return {"n": 0, "candidate": None, "incumbent": None, "diff": None, "se": None}
    d = x - y
    se = float(d.std(ddof=1) / math.sqrt(n)) if n > 1 else None
    return {"n": n, "candidate": float(x.mean()), "incumbent": float(y.mean()), "diff": float(d.mean()), "se": se}


def _kind(it: Item, i: int) -> str:
    return it.decision.options[i].kind


def measure(items: Sequence[Item], cand: Sequence[Sequence[float | None]], inc: Sequence[Sequence[float | None]]) -> dict[str, Any]:
    """The shard measures of a candidate against an incumbent on the same decisions (plain data, no human leg)."""
    tc = [top_index(v) for v in cand]
    ti = [top_index(v) for v in inc]
    idx = {s: [k for k, it in enumerate(items) if it.stratum == s] for s in QUOTAS}
    out: dict[str, Any] = {"decisions": {s: len(v) for s, v in idx.items()}}

    def rate(ix: list[int], pred: Callable[[Item, int], bool]) -> dict[str, Any]:
        return paired([pred(items[k], tc[k]) for k in ix], [pred(items[k], ti[k]) for k in ix])

    heal = lambda it, i: _kind(it, i) == "rest.heal"  # noqa: E731
    skip = lambda it, i: _kind(it, i) == "card.skip"  # noqa: E731
    node = lambda t: (lambda it, i: it.decision.options[i].node[2] == t)  # noqa: E731
    out["heal_rate"] = rate(idx["rest"], heal)
    out["rest_rate"] = rate(idx["map_rest"], node(NODE_REST))
    out["skip_rate"] = rate(idx["card_reward"], skip)
    elite = [k for k, it in enumerate(items) if it.stratum.startswith("map") and any(o.node[2] == NODE_ELITE for o in it.decision.options)
             and any(o.node[2] != NODE_ELITE for o in it.decision.options)]
    out["elite_rate"] = rate(elite, node(NODE_ELITE))
    agree = {}
    for s, ix in idx.items():
        agree[s] = paired([float(tc[k] == ti[k]) for k in ix], [1.0] * len(ix))["candidate"] if ix else None
    out["top_agreement"] = agree
    alls = [k for k in range(len(items))]
    out["top_agreement_all"] = float(np.mean([tc[k] == ti[k] for k in alls])) if alls else None
    # the heal rate by act (act 1 is the one a run's first boss depends on)
    out["heal_rate_act1"] = rate([k for k in idx["rest"] if items[k].act == 0], heal)
    out["rest_rate_act1"] = rate([k for k in idx["map_rest"] if items[k].act == 0], node(NODE_REST))
    return out


# ------------------------------------------------------------------------------------------------ the human leg


def human_leg(cand_npz: Path | str, inc_npz: Path | str, runs_root: Path | str, out_json: Path | str, *, python: str | None = None,
              max_runs: int = HUMAN_MAX_RUNS, timeout_s: float = 3600.0) -> dict[str, Any]:
    """The candidate's and the incumbent's agreement with the human choice on the held-out fifth, from the per-decision human exam
    (`tools/boss_entry_exam.py agreement` in the Spireline repo, run as a separate process at BelowNormal priority). Aggregates only:
    counts and rates over the fifth's card-reward and rest-site decisions, paired by decision. Returns the tool's record."""
    if not HUMAN_TOOL.is_file():
        raise FileNotFoundError(f"the human exam tool is not at {HUMAN_TOOL.name}; set SPIRELINE_HUMAN_EXAM_TOOL")
    cmd = [python or sys.executable, "-I", str(HUMAN_TOOL), "agreement", "--candidate", str(cand_npz), "--incumbent", str(inc_npz),
           "--runs-root", str(runs_root), "--corpus", str(HUMAN_CORPUS), "--exam-ids", str(HUMAN_FIFTH_IDS), "--max-runs", str(max_runs),
           "--fork", str(FORK_ROOT), "--out", str(out_json)]
    done = subprocess.run(cmd, timeout=timeout_s, capture_output=True, text=True)  # noqa: S603
    if done.returncode != 0:
        raise RuntimeError(f"the human exam tool exited {done.returncode}: {(done.stderr or '').strip()[-400:]}")
    return json.loads(Path(out_json).read_text(encoding="utf-8"))


# ------------------------------------------------------------------------------------------------ the rule


def judge(shard: dict[str, Any], human: dict[str, Any] | None, *, margins: dict[str, float] | None = None,
          human_z: float = HUMAN_Z) -> dict[str, Any]:
    """The rule on the measures: the checks, each with its figure and limit, and `pass` (every check passes).

    A rate check fails when the candidate's rate is more than its margin below the incumbent's. The human check fails when the
    candidate's agreement is below the incumbent's by more than `human_z` standard errors of the paired difference (when the
    standard error is zero, by any amount); a measure over fewer than `HUMAN_MIN_DECISIONS` decisions is reported and cannot fail.
    A missing human leg is a missing check: `pass` is False."""
    margins = dict(MARGIN_POINTS if margins is None else margins)
    checks = []
    for name in ("rest_rate", "heal_rate"):
        m = shard[name]
        ok = m["n"] > 0 and (m["incumbent"] - m["candidate"]) <= margins[name] + 1e-12
        checks.append({"name": name, "candidate": m["candidate"], "incumbent": m["incumbent"], "diff": m["diff"], "limit": -margins[name],
                       "n": m["n"], "pass": bool(ok)})
    if human is None:
        checks.append({"name": "human_agreement", "pass": False, "why": "the human leg did not run"})
    else:
        for name in ("card_reward", "rest"):
            m = (human.get("measures") or {}).get(name) or {}
            n = int(m.get("n") or 0)
            if n < HUMAN_MIN_DECISIONS or m.get("diff") is None:
                checks.append({"name": f"human_agreement_{name}", "n": n, "pass": True,
                               "why": f"under {HUMAN_MIN_DECISIONS} decisions: reported only"})
                continue
            se = m.get("se") or 0.0
            limit = -human_z * se
            checks.append({"name": f"human_agreement_{name}", "candidate": m["candidate"], "incumbent": m["incumbent"], "diff": m["diff"],
                           "se": se, "limit": limit, "n": n, "pass": bool(m["diff"] >= limit - 1e-12)})
    return {"pass": all(c["pass"] for c in checks), "checks": checks, "rule": RULE_TEXT,
            "margins": {**margins, "human_z": human_z}}


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def write_json(path: Path, obj: Any) -> None:
    Path(path).write_text(json.dumps(obj, indent=1) + "\n", encoding="utf-8")


def run_check(candidate_npz: Path | str, incumbent_npz: Path | str, runs_root: Path | str, out_dir: Path | str, *,
              exclude_seeds: Iterable[str] = (), human: Callable[..., dict] | None = human_leg, w: float = W_WON,
              set_doc: dict[str, Any] | None = None, margins: dict[str, float] | None = None, human_z: float = HUMAN_Z,
              items: Sequence[Item] | None = None) -> dict[str, Any]:
    """The whole check: the fixed set (read from `out_dir/ranking_set.json` when it is there and matches, else built and written),
    both heads over it, the human leg, the rule. Writes `ranking_set.json` and `ranking.json` into `out_dir` and returns the record
    (`pass`, `status`, the measures, the checks). `human` None skips the human leg (the check then cannot pass)."""
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cand_npz, inc_npz = Path(candidate_npz), Path(incumbent_npz)
    if items is None:
        exclude_seeds = list(exclude_seeds)
        if set_doc is None and (out_dir / SET_FILE).is_file():
            stored = json.loads((out_dir / SET_FILE).read_text(encoding="utf-8"))
            set_doc = stored if set_current(stored, exclude_seeds) else None  # another seed, pool or window: drawn again
        if set_doc is not None:
            items = load_set(set_doc, runs_root)
        else:
            set_doc, items = build_set(runs_root, exclude_seeds=exclude_seeds)
        write_json(out_dir / SET_FILE, set_doc)
    cand_model, inc_model = load_head(cand_npz, w), load_head(inc_npz, w)
    cand_vals, inc_vals = rank(items, cand_model), rank(items, inc_model)
    shard = measure(items, cand_vals, inc_vals)
    human_rec = None
    if human is not None:
        human_rec = human(cand_npz, inc_npz, runs_root, out_dir / "ranking_human.json")
    verdict = judge(shard, human_rec, margins=margins, human_z=human_z)
    rec = {"status": "ok", "pass": verdict["pass"], "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
           "candidate": {"sha256": file_sha256(cand_npz)}, "incumbent": {"sha256": file_sha256(inc_npz)}, "w": w,
           "set": {"file": SET_FILE, "digest": (set_doc or {}).get("digest"), "seed": (set_doc or {}).get("seed"),
                   "generations": (set_doc or {}).get("generations"), "counts": (set_doc or {}).get("counts"),
                   "runs_excluded_in_fit_window": (set_doc or {}).get("runs_excluded_in_fit_window")},
           "shard": shard, "human": human_rec, "checks": verdict["checks"], "rule": verdict["rule"], "margins": verdict["margins"],
           "wall_s": round(time.perf_counter() - t0, 1)}
    write_json(out_dir / RECORD_FILE, rec)
    return rec


def not_run(why: str) -> dict[str, Any]:
    """A ranking record for a check that could not run: it fails closed (`pass` False)."""
    return {"status": "not_run", "pass": False, "why": why, "rule": RULE_TEXT}


def no_incumbent() -> dict[str, Any]:
    """A ranking record for a first head: there is nothing to rank against, so the check does not apply (`pass` True, flagged)."""
    return {"status": "no_incumbent", "pass": True, "why": "no incumbent head to rank against", "rule": RULE_TEXT}
