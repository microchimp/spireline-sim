"""The round's data layer (docs/rl/ROUND.md): the config, the label records, the player's decisions rebuilt from the run shards.

Pure numpy; torch and the native library are never imported here. Nothing in this file writes under the agent or the lab folders.
"""

from __future__ import annotations

import copy
import glob
import gzip
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

REF = Path("D:/spireline/referee")
LABEL_SCHEMA = 2  # ROUND.md "labels file": hindsight's record (LABEL_VERSION 1) plus the schedule, the heldout flag and the clear winner

# Every weight, lambda and strength of the round is here and nowhere else (the brief: never a constant in code). Defaults are the memo
# 3.4 / 3.5 figures where the memo gives one; the others are marked in docs/rl/ROUND.md as the builder's, for the weights call.
DEFAULTS: dict[str, Any] = {
    "round": "round1",
    "paths": {"round_dir": (REF / "backlog" / "round1" / "run").as_posix(), "lab_runs": (REF / "LAB" / "runs").as_posix(),
              "agent_root": (REF / "agent").as_posix(), "lab_tree": "D:/spireline/fork/wt-gym-loop-run",
              "human_prior_dir": (REF / "human-prior").as_posix(), "human_decisions": "decisions-v1",
              "prior_weights": (REF / "human-prior" / "pi_h-v0" / "weights.npz").as_posix(),
              "gpu_python": "D:/spireline/venv-gpu/Scripts/python.exe", "emu_python": "D:/spireline/src/sts2/.venv/Scripts/python.exe"},
    "play": {"runs": 2000, "explore_tau": 0.1, "explore_eps": 0.02, "config": "a3-v1", "procs": 16},
    "label": {"k": 2, "decisions": 1200, "per_run": 2, "procs": 6, "max_options": 6, "own_schedule": True, "order_seed": 1,
              "heldout_fraction": 0.2, "task_cap_s": 3600.0,
              "quota": {"card/a1": 450, "shop/a1": 300, "rest/a1": 180, "route/a1": 150, "card/a2": 60, "shop/a2": 30, "route/a2": 20, "rest/a2": 10}},
    "prior": {"lambda": 0.5, "lambda_rule": "fixed", "lambda_grid": [0.0, 0.25, 0.5, 1.0],
              "weights": {"human": 1.0, "agent": 0.5, "label": 1.0}, "kl_beta": 1.0, "label_gap_se": 2.0,
              "lr": 1e-4, "epochs": 8, "batch": 512, "patience": 3, "weight_decay": 1e-4, "max_agent_decisions": 40000, "seed": 20261108,
              "human_fifths_train": [1, 2, 3], "human_fifth_val": 4, "kinds": ["card", "ancient", "rest", "route"]},
    "judgement": {"device": "cuda", "human_weights": [0.25, 1.0, 4.0], "seeds": [1, 2, 3], "epochs": 6, "lr": 3e-4, "batch": 1024, "beta": 0.0,
                  "fight_net": "keep", "end_hp_weight": 0.25, "heldout_fraction": 0.1, "ranking_check": True, "weight_decay": 1e-4,
                  "human_kinds": ["card", "ancient", "route"], "threads": 4},
    "smoke": {"seeds": 200, "seeds_file": (REF / "lab" / "seeds" / "R1-smoke-{date}.json").as_posix(), "keep_rule": "v2", "procs": 16},
    "block": {"seeds": 3000, "seeds_file": (REF / "lab" / "seeds" / "R1-block-{date}.json").as_posix(), "keep_rule": "v2", "procs": 16,
              "batch_seeds": 300},
    "candidate": {"tag": "round1", "target_generation": 14, "marker": False, "base": "gen11/candidate", "incumbent": "current.json",
                  "why": "D55 addendum 11 / memo row 5: the round's deck refit (prior on labels and blended choices, judgement on the new fighter's runs with human rows)"},
}


def _merge(base: dict, over: dict, path: str = "") -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k.startswith("_"):
            continue
        if k not in base:
            raise ValueError(f"round config: unknown key {path}{k}")
        if isinstance(base[k], dict) and isinstance(v, dict) and k not in ("quota", "weights"):
            out[k] = _merge(base[k], v, path + k + ".")
        else:
            out[k] = copy.deepcopy(v)
    return out


def load_config(path: str | Path | None = None, **sections: dict) -> dict:
    """DEFAULTS with the JSON file at `path` laid over them, then keyword sections (tests); an unknown key is an error."""
    cfg = copy.deepcopy(DEFAULTS)
    if path:
        cfg = _merge(cfg, json.loads(Path(path).read_text(encoding="utf-8")))
    if sections:
        cfg = _merge(cfg, sections)
    validate(cfg)
    return cfg


def validate(cfg: dict) -> None:
    p, j, lab = cfg["prior"], cfg["judgement"], cfg["label"]
    if p["lambda_rule"] not in ("fixed", "labels"):
        raise ValueError("prior.lambda_rule is 'fixed' (use prior.lambda) or 'labels' (the best of lambda_grid on held-out labels)")
    if min(p["weights"].values()) < 0 or p["kl_beta"] < 0:
        raise ValueError("prior weights and kl_beta are non-negative")
    if j["fight_net"] != "keep":
        raise ValueError("judgement.fight_net: only 'keep' is built (the incumbent's FightNet fills H for self and human rows alike)")
    if not (0 < lab["heldout_fraction"] < 1) or lab["k"] < 1:
        raise ValueError("label.heldout_fraction is in (0, 1) and label.k >= 1")


def config_text() -> str:
    return json.dumps(DEFAULTS, indent=1, sort_keys=False) + "\n"


def heldout(seed: str, fraction: float, salt: str = "round1-heldout") -> bool:
    """Whether a run (by its seed) is in the held-out part: a stable hash, so labels, agent rows and the judgement agree."""
    import hindsight  # noqa: PLC0415

    return hindsight.uniform(salt, str(seed)) < fraction


# --------------------------------------------------------------------------------------------------------- labels


def label_records(labels_dir: Path | str) -> list[dict]:
    """The usable label records (not refused, with a centred value A) of a label folder, one per decision id."""
    import hindsight  # noqa: PLC0415

    seen: dict[str, dict] = {}
    for r in hindsight.read_labels(Path(labels_dir)):
        if not r.get("refused") and r.get("A"):
            seen[r["id"]] = r
    return [seen[k] for k in sorted(seen)]


def pooled_pair_sd(recs: Iterable[dict]) -> float | None:
    """The pooled standard deviation (floors) of one draw's difference between two options, over the records: the noise of a label."""
    import hindsight  # noqa: PLC0415

    vs: list[tuple[float, int]] = []
    for r in recs:
        g = hindsight.decision_g(r)
        if g:
            vs += hindsight.paired_variances(g)
    v = hindsight.pooled(vs)
    return None if v is None else math.sqrt(v)


def clear_winner(rec: dict, sd: float | None, z: float) -> int | None:
    """The option (screen index) a label names as clearly best: the best played option's mean exceeds the next best by more than
    `z` standard errors of a paired difference (`sd` / sqrt(K), pooled). None when there is no such gap or no noise figure."""
    a = {int(k): float(v) for k, v in rec["A"].items()}
    if len(a) < 2 or sd is None:
        return None
    order = sorted(a, key=lambda i: -a[i])
    se = sd / math.sqrt(max(1, int(rec["k"])))
    return order[0] if a[order[0]] - a[order[1]] > z * se else None


# --------------------------------------------------------------------------------------------------------- runs


def index_runs(runs_root: Path | str, job: str, seeds: set[str]) -> dict[str, dict]:
    """The recorded runs (shard lines) of `job` whose seed is in `seeds`."""
    found: dict[str, dict] = {}
    for p in sorted((Path(runs_root) / job / "shards").glob("r*.jsonl.gz")):
        with gzip.open(p, "rt", encoding="utf-8") as fh:
            for line in fh:
                if line.strip():
                    rec = json.loads(line)
                    if str(rec.get("seed")) in seeds:
                        found[str(rec["seed"])] = rec
    return found


def iter_runs(runs_root: Path | str, jobs: Sequence[str]) -> Iterable[tuple[str, dict]]:
    for job in jobs:
        for p in sorted(glob.glob(str(Path(runs_root) / job / "shards" / "r*.jsonl.gz"))):
            with gzip.open(p, "rt", encoding="utf-8") as fh:
                for line in fh:
                    if line.strip():
                        yield job, json.loads(line)


@dataclass
class Item:
    """A decision rebuilt from a run shard: the judgement's input rows of every option."""

    rows: np.ndarray  # (n, 1863)
    screen: str
    act: int
    seed: str
    chosen: int = -1
    played: list[int] = field(default_factory=list)
    A: dict[int, float] = field(default_factory=dict)
    k: int = 0
    winner: int | None = None  # a label's clear winner (screen index), when its gap is over the SE bound
    kinds: list[str] = field(default_factory=list)  # each option's kind (the skip option, the rest groups)
    groups: list[int] = field(default_factory=list)


def _decision_rows(rec: dict, entry: dict, nets: Any) -> tuple[Any, np.ndarray, str, Any] | None:
    from human_prior_infer import option_rows  # noqa: PLC0415
    import imitation_chooser as ic  # noqa: PLC0415
    from run_train import shards  # noqa: PLC0415

    dec = shards.entry_decision(entry)
    screen = ic.screen_of(dec)
    if screen not in ("card", "rest"):
        return None
    view = dec.view or shards.state_view(rec, entry)
    return dec, option_rows(nets, view, list(dec.options)), screen, view


def agent_items(runs_root: Path | str, jobs: Sequence[str], nets: Any, *, keep: float, hold: bool | None, fraction: float,
                limit: int, order_salt: str = "round1-agent") -> list[Item]:
    """The player's own deck-screen choices from its recorded runs: one Item per card or rest decision, `chosen` the option the
    blended chooser took. `hold` True keeps held-out runs only, False the others, None all; `keep` is the share of decisions
    sampled (a stable hash of run and ordinal); `limit` caps the count."""
    import hindsight  # noqa: PLC0415

    out: list[Item] = []
    for _job, rec in iter_runs(runs_root, jobs):
        seed = str(rec.get("seed"))
        if hold is not None and heldout(seed, fraction) != hold:
            continue
        for i, entry in enumerate(rec.get("decisions", [])):
            if hindsight.uniform(order_salt, seed, i) >= keep:
                continue
            got = _decision_rows(rec, entry, nets)
            if got is None or entry.get("chosen") is None:
                continue
            dec, rows, screen, view = got
            out.append(Item(rows=rows.astype(np.float32), screen=screen, act=int(view.act_index), seed=seed, chosen=int(entry["chosen"])))
            if len(out) >= limit:
                return out
    return out


def label_items(recs: Sequence[dict], runs_root: Path | str, nets: Any, winner_sd: float | None = None,
                winner_z: float = 2.0) -> tuple[list[Item], dict[str, int]]:
    """Label records rebuilt to Items (the screen, the rows of every option, the played shortlist and A); a record whose screen cannot
    be rebuilt to the label's own options is counted by reason and left out."""
    import imitation_chooser as ic  # noqa: PLC0415
    from run_train import shards  # noqa: PLC0415

    stats: dict[str, int] = defaultdict(int)
    by_job: dict[str, list[dict]] = defaultdict(list)
    for r in recs:
        by_job[r["run"]["job"]].append(r)
    items: list[Item] = []
    for job, rs in by_job.items():
        runs = index_runs(runs_root, job, {str(r["run"]["seed"]) for r in rs})
        for r in rs:
            rec = runs.get(str(r["run"]["seed"]))
            if rec is None:
                stats["run_not_found"] += 1
                continue
            try:
                entry = rec["decisions"][r["ordinal"]]
            except IndexError:
                stats["ordinal_out_of_range"] += 1
                continue
            got = _decision_rows(rec, entry, nets)
            if got is None:
                stats["not_a_prior_screen"] += 1
                continue
            dec, rows, screen, _view = got
            mine = [shards.option_to_json(o) for o in dec.options]
            theirs = [{k: v for k, v in o.items() if k != "idx"} for o in r["options"]]
            if mine != theirs or entry["type"] != r["type"]:
                stats["options_differ"] += 1
                continue
            played = [int(i) for i in r["played"]]
            if len(played) < 2:
                stats["one_played"] += 1
                continue
            items.append(Item(rows=rows.astype(np.float32), screen=screen, act=int(r["act"]), seed=str(r["run"]["seed"]),
                              chosen=int(r["taken"]), played=played, A={int(k): float(v) for k, v in r["A"].items()}, k=int(r["k"]),
                              kinds=[o.kind for o in dec.options], groups=[ic.rest_group(o) for o in dec.options],
                              winner=clear_winner(r, winner_sd, winner_z)))
            stats["rebuilt"] += 1
    return items, dict(stats)


# --------------------------------------------------------------------------------------------------------- the label leg


def mean_se(xs: Sequence[float]) -> list[float | None]:
    a = np.asarray(xs, dtype=float)
    if not len(a):
        return [None, None]
    return [float(a.mean()), float(a.std(ddof=1) / math.sqrt(len(a))) if len(a) > 1 else None]


def blend_pick(item: Item, values: Sequence[float], logits: Sequence[float], lam: float) -> int:
    """The option (screen index) the blended chooser takes among the label's played options: argmax V + lam log pi_h, the chooser's own
    rules (`imitation_chooser.blend_pick` for cards, `blend_rest` for rests). lam = inf is the prior alone."""
    import imitation_chooser as ic  # noqa: PLC0415
    from human_prior_infer import log_softmax  # noqa: PLC0415

    pl = item.played
    v = [float(values[i]) for i in pl]
    lg = [float(logits[i]) for i in pl]
    if item.screen == "rest":
        return pl[ic.blend_rest(v, lg, [item.groups[i] for i in pl], lam).index]
    return pl[ic.blend_pick(v, log_softmax(np.asarray(lg)), lam).index]


def leg_picks(items: Sequence[Item], arms: dict[str, tuple[Sequence[Sequence[float]], Sequence[Sequence[float]], float]]) -> dict:
    """The held-out label leg. `arms[name] = (values per item, logits per item, lambda)`: the picks of each arm among the played options,
    their mean centred label value A, and every arm's difference from the first, with the standard error and 95% interval. The
    player's own pick of the run and the best option (oracle) are reported beside them. Cells under 20 decisions are not shown."""
    def block(sel: list[int]) -> dict | None:
        if len(sel) < 20:
            return None
        res: dict[str, Any] = {"n": len(sel)}
        picks = {n: np.array([items[i].A[blend_pick(items[i], v[i], l[i], lam)] for i in sel]) for n, (v, l, lam) in arms.items()}
        for n, x in picks.items():
            res[n] = mean_se(x)
        first = next(iter(picks))
        for n, x in picks.items():
            if n != first:
                d = x - picks[first]
                m, se = mean_se(d)
                res[f"{n}_minus_{first}"] = {"mean": m, "se": se, "lo95": None if se is None else m - 1.96 * se,
                                             "hi95": None if se is None else m + 1.96 * se}
        res["players_own_pick"] = mean_se([items[i].A.get(items[i].chosen, 0.0) for i in sel])
        res["best_option"] = mean_se([max(items[i].A.values()) for i in sel])
        return res

    out: dict[str, Any] = {}
    for name, sel in (("all", list(range(len(items)))), ("card", [i for i, it in enumerate(items) if it.screen == "card"]),
                      ("rest", [i for i, it in enumerate(items) if it.screen == "rest"]),
                      ("act1", [i for i, it in enumerate(items) if it.act == 0]), ("act2", [i for i, it in enumerate(items) if it.act >= 1])):
        b = block(sel)
        if b is not None:
            out[name] = b
    return out
