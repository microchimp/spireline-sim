"""The expert-iteration generation loop (docs/PLAN.md section 5 Lane A, docs/DESIGN.md 5.3-5.4).

    python rl/gen0.py emu --stream JOB --config a3-v1 --record --agent-root AGENT --out JOBDIR   # self-play with samples
    python rl/gen0.py train --gen N --shards-root RUNS --agent-root AGENT                         # -> AGENT/gen<N+1>/candidate
    python rl/gen0.py promote --cand AGENT/gen<N+1>/candidate --base AGENT/gen<N>/...             # E1 block, both arms

A *generation* is the agent that plays: generation 0 is the search with no network and the rules policy outside fights;
generation N+1 is generation N's candidate once `promote` accepts it. A self-play block records, per run,

- the run in `run_train`'s RunRecord format (decisions with the run state and options, fights, outcome), tagged with the
  generation that played it: `run_train.train.fit` trains V_run (DMC target on the final floor) and FightNet from it;
- per searched combat decision, in `prior_loop`'s dataset layout (`DATASET_KEYS`): the token encoding of the root,
  the legal mask, the search's root visits (the improved-policy target `train_combat.train` reads) and the root's mean
  values (`qvals`, for a completed-Q target), the fight's final HP (the value label), and `fights` = the run's index in
  its part so `train_combat`'s validation split by "fight" is a split by RUN.

The live search has no Gumbel root (A3 report: no Gumbel knobs exist), so the target is the root visit distribution.

Native code is imported inside the functions that need it; the maths and the shard format import none.
"""

from __future__ import annotations

import contextlib
import dataclasses
import io
import json
import math
import os
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGENT_ROOT = ROOT.parents[1] / "referee" / "agent"
GEN_WINDOW = 3  # DESIGN 5.4: train on the last 3 generations
BUDGET_RUNS = 2000  # self-play runs per generation before its train marker
MAX_FIGHTS = 6400  # DESIGN 5.4: fights sampled for the combat net
PART_RUNS = 50
ACT1_LAST_FLOOR = 17  # the Act 1 boss is floor 17; a run past it killed the boss
ALPHA = 0.01
WIN_GATE = 0.10  # wins join the co-primaries once both arms reach this rate
BOOT_B = 20000
ENV_GEN_DIR = "SPIRELINE_GEN_DIR"
PRIOR_SEARCH = {"c_puct": 0.2, "prior_min_visits": 3}  # prior_loop's PriorConfig defaults

TOKEN_KEYS = ("types", "ids", "owners", "feats")
ROW_KEYS = ("lengths", "labels", "encounters", "deck_sizes", "relic_counts", "visits", "legal", "returns",
            "qvals", "sims", "fight_ix")


def utc_now() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, doc: Any) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


# ------------------------------------------------------------------ the agent a generation directory names


@dataclass(frozen=True)
class Agent:
    """What plays: the combat prior (if the directory has one) and the macro policy outside fights."""

    path: str | None = None
    generation: int = 0
    prior_dir: str | None = None
    macro: str = "rules"  # "rules" or "vrun" (one-step lookahead on V_run, run_train.policy.LookaheadPolicy)
    macro_npz: str | None = None
    tau: float = 0.0
    eps: float = 0.0
    W: float = 53.0  # the value of a won run (run_train.vocab.WON_FLOOR)

    def search_extra(self) -> dict[str, Any]:
        """`live_search.search_live` keyword arguments that switch the tree to PUCT with the prior; {} for none."""
        if not self.prior_dir:
            return {}
        return dict(_champion_settings(self.prior_dir))

    def macro_policy(self):
        from run_train import selfplay  # noqa: PLC0415

        if self.macro == "vrun":
            from run_train.policy import LookaheadPolicy  # noqa: PLC0415

            return LookaheadPolicy(self.macro_npz, self.tau, eps=self.eps, W=self.W)
        return selfplay.RulesPlayer()

    def describe(self) -> dict[str, Any]:
        sha = None
        if self.prior_dir:
            with contextlib.suppress(OSError, ValueError):
                sha = read_json(Path(self.prior_dir) / "model.json").get("policy_sha256")
        tail = "/".join(Path(self.path).parts[-2:]) if self.path else None
        return {"generation": self.generation, "dir": tail, "prior": bool(self.prior_dir), "prior_sha256": sha,
                "macro": self.macro}


@lru_cache(maxsize=8)
def _champion_settings(prior_dir: str) -> tuple[tuple[str, Any], ...]:
    import combat_prior  # noqa: PLC0415 -- native

    champion = combat_prior.find(prior_dir)
    if champion is None:
        raise FileNotFoundError(f"no combat prior (model.json and combat_policy.pt) in {Path(prior_dir).name}")
    return tuple(champion.live_settings().items())


def load_agent(gen_dir: str | Path | None = None, agent_root: str | Path | None = None, macro: str | None = None) -> Agent:
    """The agent a `--gen-dir` names. A `current.json` file (or `agent_root/current.json`) points at the promoted one;
    no argument is generation 0."""
    if gen_dir is None and agent_root is not None and (Path(agent_root) / "current.json").is_file():
        gen_dir = Path(agent_root) / "current.json"
    if gen_dir is None:
        return Agent()
    p = Path(gen_dir)
    if p.is_file():
        p = Path(read_json(p)["dir"])
    meta = read_json(p / "agent.json") if (p / "agent.json").is_file() else {}
    m = re.fullmatch(r"gen(\d+)", p.name) or re.fullmatch(r"gen(\d+)", p.parent.name)
    generation = int(meta.get("generation", m.group(1) if m else 0))
    prior = str(p) if (p / "model.json").is_file() and (p / "combat_policy.pt").is_file() else None
    mode = macro or meta.get("macro", "rules")
    npz = p / "run_value.npz"
    if mode == "vrun" and not npz.is_file():
        raise FileNotFoundError(f"macro vrun needs run_value.npz in {p.name}")
    return Agent(str(p), generation, prior, mode, str(npz) if mode == "vrun" else None,
                 float(meta.get("tau", 0.0)), float(meta.get("eps", 0.0)))


def active_agent_extra() -> dict[str, Any]:
    """`search_configs.SearchConfig.search_kwargs` adds this: the prior of the agent named by SPIRELINE_GEN_DIR (score-real's
    lockstep player builds its search from a config alone), nothing when the variable is unset."""
    d = os.environ.get(ENV_GEN_DIR)
    return load_agent(d).search_extra() if d else {}


# ------------------------------------------------------------------ per-run recording (inside the worker)


class CombatSamples:
    """The searched combat decisions of one run, in `prior_loop`'s layout."""

    def __init__(self) -> None:
        self.rows: list[dict[str, Any]] = []
        self._open: list[int] = []
        self.fight = 0
        self.max_hp = 0

    def add(self, tokens, visits: np.ndarray, qvals: np.ndarray, legal: np.ndarray, encounter: int, deck: int, relics: int,
            sims: int) -> None:
        self._open.append(len(self.rows))
        self.rows.append({"tokens": tokens, "visits": visits, "qvals": qvals, "legal": legal, "encounter": int(encounter),
                          "deck": int(deck), "relics": int(relics), "sims": int(sims), "fight": self.fight, "label": 0})

    def end_fight(self, end_hp: int, max_hp: int) -> None:
        for i in self._open:
            self.rows[i]["label"] = int(end_hp)
        self._open = []
        self.fight += 1
        self.max_hp = max(self.max_hp, int(max_hp))

    def arrays(self) -> dict[str, np.ndarray] | None:
        if not self.rows:
            return None
        r = self.rows
        n_act = len(r[0]["visits"])
        return {
            "types": np.concatenate([x["tokens"].types for x in r]).astype(np.int64),
            "ids": np.concatenate([x["tokens"].ids for x in r]).astype(np.int64),
            "owners": np.concatenate([x["tokens"].owners for x in r]).astype(np.int64),
            "feats": np.concatenate([x["tokens"].feats for x in r]).astype(np.float32),
            "lengths": np.array([len(x["tokens"]) for x in r], dtype=np.int64),
            "labels": np.array([x["label"] for x in r], dtype=np.int64),
            "encounters": np.array([x["encounter"] for x in r], dtype=np.int64),
            "deck_sizes": np.array([x["deck"] for x in r], dtype=np.int64),
            "relic_counts": np.array([x["relics"] for x in r], dtype=np.int64),
            "visits": np.stack([x["visits"] for x in r]).astype(np.float32).reshape(len(r), n_act),
            "legal": np.stack([x["legal"] for x in r]).astype(np.int8),
            "returns": np.full((len(r), 1), np.nan, dtype=np.float32),  # the live search keeps no per-simulation returns
            "qvals": np.stack([x["qvals"] for x in r]).astype(np.float32),
            "sims": np.array([x["sims"] for x in r], dtype=np.int32),
            "fight_ix": np.array([x["fight"] for x in r], dtype=np.int64),
            "max_hp": np.array([self.max_hp], dtype=np.int64),
        }


def root_targets(children, n_actions: int) -> tuple[np.ndarray, np.ndarray]:
    """The search's root visit distribution over action ids, and its mean value per action.

    A move merged from identical cards (`merged_actions`) shares its visits evenly among them, and the moves aimed at
    different enemies add up under their action. Actions with no visit have value NaN.
    """
    visits = np.zeros(n_actions, dtype=np.float64)
    wsum = np.zeros(n_actions, dtype=np.float64)
    for ch in children:
        group = [ch.action, *ch.merged_actions]
        share = ch.visits / len(group)
        for a in group:
            if 0 <= a < n_actions:
                visits[a] += share
                if ch.visits and math.isfinite(ch.value_mean):
                    wsum[a] += share * ch.value_mean
    with np.errstate(invalid="ignore", divide="ignore"):
        q = np.where(visits > 0, wsum / np.where(visits > 0, visits, 1.0), np.nan)
    return visits.astype(np.float32), q.astype(np.float32)


class RunRecorder:
    """Builds one run's RunRecord (run_train.shards) and its combat samples while `gen0.play_emu_run` plays."""

    def __init__(self, seed: str, generation: int, policy_desc: dict, base_sims: int, c: float, ascension: int) -> None:
        from run_train import selfplay  # noqa: PLC0415

        self.seed, self.generation, self.policy_desc = seed, generation, policy_desc
        self.base_sims, self.c, self.ascension = base_sims, c, ascension
        self.run = selfplay._Run()  # noqa: SLF001 -- the record's own accumulator (maps, states)
        self.samples = CombatSamples()
        self.fight: dict[str, Any] | None = None
        self.started = time.perf_counter()

    def decision(self, view, decision, pick) -> None:
        from run_train import shards  # noqa: PLC0415

        run = self.run
        run.decisions.append({
            "i": len(run.decisions), "type": decision.type, "floor": view.floor, "state": run.state(view),
            "options": [shards.option_to_json(o) for o in decision.options], "chosen": pick.index,
            "values": None if pick.values is None else list(pick.values), "p_chosen": pick.p_chosen,
            "fallback": pick.fallback, "p_floor": None if pick.p_floor is None else list(pick.p_floor)})

    def open_fight(self, session, combat) -> None:
        from run_train import vocab  # noqa: PLC0415

        run, view = self.run, session.view()
        if view.act_index != run.act_index:
            run.act_index, run.normal_fights = view.act_index, 0
        if combat.node_type == vocab.NODE_MONSTER:
            run.normal_fights += 1
        weak = combat.node_type == vocab.NODE_MONSTER and run.normal_fights <= vocab.weak_fights(view.act_region)
        self.fight = {"floor": view.floor, "node_type": combat.node_type, "encounter": combat.encounter_id, "weak": weak,
                      "sims": combat.sims, "state": run.state(view), "hp_before": view.hp, "decisions": 0,
                      "t": time.perf_counter(), "max_hp": view.max_hp, "deck": len(view.deck), "relics": len(view.relics)}

    def close_fight(self, session, won: bool) -> None:
        f = self.fight
        if f is None:
            return
        hp_after = max(0, int(session.info["hp"])) if won else 0
        self.run.fights.append({"floor": f["floor"], "node_type": f["node_type"], "encounter": f["encounter"], "weak": f["weak"],
                                "sims": f["sims"], "state": f["state"], "hp_before": f["hp_before"], "hp_after": hp_after,
                                "won": bool(won), "decisions": f["decisions"], "seconds": time.perf_counter() - f["t"]})
        self.samples.end_fight(hp_after if won else 0, f["max_hp"])
        self.fight = None

    def sample(self, root, res, tokens) -> None:
        """One searched decision: `tokens` were encoded from `root` before the search, `res` is its result."""
        from combat_tokens import MAX_ACTIONS  # noqa: PLC0415
        from sts2_gym import native  # noqa: PLC0415

        f = self.fight
        f["decisions"] += 1
        if len(res.children) < 2:
            return  # a forced move teaches the policy nothing
        mask = native.valid_actions(root.handle, MAX_ACTIONS)
        legal = np.array([1 if mask[a] else 0 for a in range(MAX_ACTIONS)], dtype=np.int8)
        visits, q = root_targets(res.children, MAX_ACTIONS)
        self.samples.add(tokens, visits, q, legal, f["encounter"], f["deck"], f["relics"], res.sims)

    def finish(self, session, moves: int, won: bool, steps_cap_hit: bool = False) -> dict[str, Any]:
        from run_train import FEATURE_VERSION, shards  # noqa: PLC0415
        from run_train.vocab import WON_FLOOR  # noqa: PLC0415

        self.run.moves = moves
        info, stats = session.info, session.stats
        return {
            "schema": shards.SCHEMA_VERSION, "feature_version": FEATURE_VERSION, "seed": self.seed,
            "generation": self.generation, "policy": self.policy_desc,
            "combat": {"budget": str(session.budget), "c": self.c, "crn_fraction": 1.0},
            "ascension": self.ascension,
            "outcome": {"final_floor": WON_FLOOR if won else info["floor"], "won": bool(won),
                        "acts_cleared": session.acts_cleared, "died": bool(session.died), "end_hp": max(0, info["hp"]),
                        "end_max_hp": info["max_hp"], "truncated": steps_cap_hit},
            "maps": self.run.maps, "decisions": self.run.decisions, "fights": self.run.fights,
            "stats": {"unknown_decisions": dict(stats.unknown_decisions), "forced_skips": dict(stats.forced_skips),
                      "potion_left_full_belt": stats.potion_left_full_belt, "steps": session.steps + moves,
                      "seconds": time.perf_counter() - self.started}}


# ------------------------------------------------------------------ shards of a self-play block


def write_part(path: Path, runs: list[tuple[str, dict | None]], generation: int) -> int:
    """Write the combat samples of `runs` (seed, arrays-or-None) as one npz in `prior_loop`'s layout; return the sample count.
    `fights` is the run's index in the part (so a split by that column is a split by run)."""
    path = Path(path)
    have = [(i, a) for i, (_, a) in enumerate(runs) if a is not None]
    out: dict[str, np.ndarray] = {}
    for key in TOKEN_KEYS + ROW_KEYS:
        if have:
            out[key] = np.concatenate([a[key] for _, a in have])
    n = int(out["lengths"].size) if have else 0
    if have:
        out["fights"] = np.concatenate([np.full(a["lengths"].size, i, dtype=np.int64) for i, a in have])
        out["max_hp"] = np.array([max(int(a["max_hp"][0]) for _, a in have)], dtype=np.int64)
    out["seeds"] = np.array([s for s, _ in runs])
    out["generation"] = np.array([generation], dtype=np.int64)
    out["n_runs"] = np.array([len(runs)], dtype=np.int64)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.stem + ".tmp.npz")
    np.savez_compressed(tmp, **out)
    os.replace(tmp, path)
    return n


def read_part(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as blob:
        return {k: blob[k] for k in blob.files}


class ShardSink:
    """The main process's writer for one self-play block: run shards (RunRecords) and combat parts, and `shards.json`,
    which names the generation that played them. The manifest is rewritten at every flush, so an interrupted block is
    still readable."""

    def __init__(self, out_dir: Path, generation: int, *, config: str, base_sims: int, ascension: int, agent: dict,
                 part_runs: int = PART_RUNS) -> None:
        from run_train import shards  # noqa: PLC0415

        self.dir = Path(out_dir) / "shards"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.generation, self.part_runs = generation, part_runs
        self.meta = {"kind": "selfplay-shards", "generation": generation, "config": config, "base_sims": base_sims,
                     "ascension": ascension, "agent": agent}
        self.writer = shards.ShardWriter(self.dir, part_runs)
        self.pending: list[tuple[str, dict | None]] = []
        self.parts: list[str] = []
        self.runs = self.samples = 0

    def add(self, record: dict, arrays: dict | None) -> None:
        assert record["generation"] == self.generation, "a shard is tagged with the generation that played it"
        self.writer.add(record)
        self.pending.append((record["seed"], arrays))
        self.runs += 1
        if len(self.pending) >= self.part_runs:
            self.flush()

    def flush(self) -> None:
        if self.pending:
            name = f"combat-g{self.generation}-{len(self.parts):04d}.npz"
            self.samples += write_part(self.dir / name, self.pending, self.generation)
            self.parts.append(name)
            self.pending = []
        self.writer.flush()
        self._manifest()

    def _manifest(self) -> None:
        from run_train import shards  # noqa: PLC0415

        write_json(self.dir / "shards.json", {**self.meta, "runs": self.runs, "samples": self.samples,
                                              "run_shards": [p.name for p in shards.shard_paths(self.dir)],
                                              "combat_parts": list(self.parts), "written": utc_now()})

    def close(self) -> dict:
        self.flush()
        return read_json(self.dir / "shards.json")


def find_manifests(root: Path) -> list[tuple[Path, dict]]:
    """Every `shards/shards.json` directly under the job directories of `root` (or under `root` itself)."""
    root = Path(root)
    found = sorted(root.glob("*/shards/shards.json")) + ([root / "shards" / "shards.json"] if (root / "shards" / "shards.json").is_file() else [])
    out = []
    for m in found:
        with contextlib.suppress(OSError, ValueError):
            out.append((m.parent, read_json(m)))
    return out


def window_manifests(root: Path, gen: int, window: int = GEN_WINDOW) -> list[tuple[Path, dict]]:
    return [(d, m) for d, m in find_manifests(root) if gen - window < int(m["generation"]) <= gen]


def runs_in_generation(root: Path, gen: int) -> int:
    return sum(int(m.get("runs", 0)) for _, m in find_manifests(root) if int(m["generation"]) == gen)


def build_training_set(manifests: list[tuple[Path, dict]], *, max_fights: int = MAX_FIGHTS, seed: int = 0) -> dict[str, Any]:
    """Merge the combat parts of the window with `prior_loop.loop.merge_datasets` and sample at most `max_fights` fights.
    Returns the arrays plus counts per generation."""
    from prior_loop.loop import merge_datasets  # noqa: PLC0415

    sources, per_gen, ixs = [], {}, []
    for d, m in manifests:
        for name in m.get("combat_parts", []):
            blob = read_part(d / name)
            if "lengths" not in blob:  # a part whose runs recorded no searched decision
                continue
            g = int(m["generation"])
            per_gen[g] = per_gen.get(g, 0) + int(blob["lengths"].size)
            ixs.append(blob["fight_ix"])
            sources.append((len(sources), d / name))
    if not sources:
        raise ValueError("no combat samples in the window")
    merged = merge_datasets(sources)  # offsets each part's run index, so a run id is unique over the window
    key = merged["fights"] * 4096 + np.concatenate(ixs)  # a fight = (run, fight_ix in the run)
    uniq, inv = np.unique(key, return_inverse=True)
    n_all = int(uniq.size)
    if n_all > max_fights:
        pick = np.random.default_rng(seed).permutation(n_all)[:max_fights]
        keep = np.isin(inv, pick)
        merged = _take_rows(merged, keep)
        kept = int(np.unique(key[keep]).size)
    else:
        kept = n_all
    merged["_fights_available"] = np.array([n_all])
    merged["_fights_kept"] = np.array([kept])
    merged["_samples_by_generation"] = np.array(sorted(per_gen.items()))
    return merged


def _take_rows(data: dict[str, np.ndarray], keep: np.ndarray) -> dict[str, np.ndarray]:
    tok = np.repeat(keep, data["lengths"])
    out = {}
    for k, v in data.items():
        if k in TOKEN_KEYS:
            out[k] = v[tok]
        elif k == "max_hp" or v.ndim == 0 or v.shape[0] != keep.size:
            out[k] = v
        else:
            out[k] = v[keep]
    return out


# ------------------------------------------------------------------ training (CPU or the GPU venv)

_EPOCH = re.compile(r"epoch\s+(\d+)\s+value ([0-9.]+)/([0-9.]+)\s+policy ([0-9.]+)/([0-9.]+)")


def parse_curves(text: str) -> list[dict]:
    return [{"epoch": int(m[1]), "value_train": float(m[2]), "value_val": float(m[3]), "policy_train": float(m[4]),
             "policy_val": float(m[5])} for m in _EPOCH.finditer(text)]


def _device(requested: str | None) -> str:
    import torch  # noqa: PLC0415

    return requested or ("cuda" if torch.cuda.is_available() else "cpu")


def train_combat_prior(data: dict[str, np.ndarray], out: Path, *, device: str | None = None, epochs: int = 16, d_model: int = 128,
                       layers: int = 2, threads: int = 6, seed: int = 0) -> dict[str, Any]:
    """Fit the policy and value heads with `train_combat.train` (prior_loop's trainer, its settings and its checkpoints)."""
    import argparse  # noqa: PLC0415

    import torch  # noqa: PLC0415
    import train_combat  # noqa: PLC0415

    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    work = out / "work"
    work.mkdir(exist_ok=True)
    npz = work / "train.npz"
    np.savez_compressed(npz, **{k: v for k, v in data.items() if not k.startswith("_")})
    dev = _device(device)
    torch.set_num_threads(threads)
    args = argparse.Namespace(epochs=epochs, value_lambda=0.0, seed=seed, bucket=5, batch=64, lr=3e-4, policy_weight=1.0,
                              d_model=d_model, layers=layers, heads=4, ablate_endturn=True, plain_cards=False, fraction=1.0,
                              device=dev)
    saved = {n: getattr(train_combat, n) for n in ("DATASET", "CHECKPOINT", "POLICY_CHECKPOINT")}
    cap = io.StringIO()
    t0, c0 = time.perf_counter(), time.process_time()
    try:
        train_combat.DATASET, train_combat.CHECKPOINT = npz, out / "combat_net.pt"
        train_combat.POLICY_CHECKPOINT = out / "combat_policy.pt"
        with contextlib.redirect_stdout(cap):
            train_combat.train(args)
    finally:
        (work / "train_combat.log").write_text(cap.getvalue(), encoding="utf-8")
        for n, v in saved.items():
            setattr(train_combat, n, v)
    text = cap.getvalue()
    (work / "train_combat.log").write_text(text, encoding="utf-8")
    # CPU tensors whatever device trained them: the searches load them on the CPU
    for name in ("combat_net.pt", "combat_policy.pt"):
        blob = torch.load(out / name, map_location="cpu", weights_only=True)
        torch.save(blob, out / name)
    from prior_loop.loop import parse_train_report  # noqa: PLC0415

    return {"device": dev, "epochs": epochs, "d_model": d_model, "layers": layers, "wall_min": (time.perf_counter() - t0) / 60,
            "cpu_min": (time.process_time() - c0) / 60, "report": parse_train_report(text), "curves": parse_curves(text),
            "samples": int(data["lengths"].size)}


def write_prior_model_json(out: Path, generation: int, probe: bool = False) -> dict:
    import combat_prior  # noqa: PLC0415

    sha = combat_prior._sha256_file(Path(out) / "combat_policy.pt")  # noqa: SLF001
    doc = {"kind": combat_prior.KIND, "iteration": generation, "vocab_sha256": combat_prior.vocab_sha256(), "policy_sha256": sha,
           "probe": probe, "search": dict(PRIOR_SEARCH)}
    write_json(Path(out) / "model.json", doc)
    return doc


def train_run_value(shard_paths: list[Path], out: Path, *, generation: int, base_sims: int, ascension: int, threads: int = 6) -> dict:
    """Fit FightNet and V_run (DMC target, `run_train.train.fit`) on the window's run records."""
    import torch  # noqa: PLC0415
    from run_train import train as rt  # noqa: PLC0415
    from run_train.engine import CombatBudget  # noqa: PLC0415

    t0, c0 = time.perf_counter(), time.process_time()
    pt = Path(out) / "run_value.pt"
    rt.fit(shard_paths, None, pt, generation=generation, combat_budget=CombatBudget(base_sims, base_sims), ascension=ascension,
           config=rt.TrainConfig(value_generations=GEN_WINDOW, threads=threads))
    meta = torch.load(pt, map_location="cpu", weights_only=True)["meta"]
    return {"wall_min": (time.perf_counter() - t0) / 60, "cpu_min": (time.process_time() - c0) / 60,
            "fight": meta.get("fight"), "value": meta.get("value"), "config": meta.get("config")}


# ------------------------------------------------------------------ ONNX


def _varint(b: bytes, i: int) -> tuple[int, int]:
    n = shift = 0
    while True:
        c = b[i]
        i += 1
        n |= (c & 0x7F) << shift
        if not c & 0x80:
            return n, i
        shift += 7


def _fields(b: bytes):
    i = 0
    while i < len(b):
        tag, i = _varint(b, i)
        no, wire = tag >> 3, tag & 7
        if wire == 0:
            v, i = _varint(b, i)
        elif wire == 2:
            ln, i = _varint(b, i)
            v, i = b[i:i + ln], i + ln
        elif wire == 1:
            v, i = b[i:i + 8], i + 8
        elif wire == 5:
            v, i = b[i:i + 4], i + 4
        else:
            raise ValueError(f"unsupported wire type {wire}")
        yield no, wire, v


def onnx_structure(path: Path) -> dict[str, Any]:
    """Read an ONNX file back without the onnx package: ModelProto -> GraphProto node count, op types, inputs, outputs."""
    model = Path(path).read_bytes()
    graph = next((v for no, w, v in _fields(model) if no == 7 and w == 2), None)
    if graph is None:
        raise ValueError("no graph in the file")
    nodes, ops, ins, outs, inits = 0, set(), [], [], 0
    for no, w, v in _fields(graph):
        if no == 1 and w == 2:
            nodes += 1
            ops.update(x.decode() for n2, w2, x in _fields(v) if n2 == 4 and w2 == 2)
        elif no == 5 and w == 2:
            inits += 1
        elif no in (11, 12) and w == 2:
            name = next((x.decode() for n2, w2, x in _fields(v) if n2 == 1 and w2 == 2), "")
            (ins if no == 11 else outs).append(name)
    inputs = [n for n in ins]
    return {"nodes": nodes, "ops": sorted(ops), "inputs": inputs, "outputs": outs, "initializers": inits,
            "ir_version": next((v for no, w, v in _fields(model) if no == 1 and w == 0), None)}


def export_onnx(module, example: tuple, input_names: list[str], output_names: list[str], dynamic: dict, path: Path) -> dict:
    """Export with torch's own exporter (offline, TorchScript path), then read the file back: with onnxruntime if it is
    installed (and compare its outputs with torch's), else with onnx, else by `onnx_structure`. A failure is returned as
    the reason; it never raises."""
    import torch  # noqa: PLC0415

    path = Path(path)
    module.eval()
    fast = torch.backends.mha.get_fastpath_enabled()
    torch.backends.mha.set_fastpath_enabled(False)  # the fused encoder layer of eval mode has no ONNX operator
    try:
        with torch.no_grad(), contextlib.redirect_stdout(io.StringIO()):
            torch.onnx.export(module, example, str(path), input_names=input_names, output_names=output_names,
                              dynamic_axes=dynamic, opset_version=17, dynamo=False)
    except Exception as e:  # noqa: BLE001
        path.unlink(missing_ok=True)
        return {"ok": False, "reason": f"export: {type(e).__name__}: {str(e)[:300]}"}
    finally:
        torch.backends.mha.set_fastpath_enabled(fast)
    doc: dict[str, Any] = {"ok": True, "file": path.name, "bytes": path.stat().st_size}
    try:
        import onnxruntime as ort  # noqa: PLC0415

        sess = ort.InferenceSession(str(path))
        feeds = {n: x.numpy() for n, x in zip(input_names, example, strict=True)}
        got = sess.run(None, feeds)
        with torch.no_grad():
            want = module(*example)
        want = want if isinstance(want, tuple) else (want,)
        doc["checked_with"] = "onnxruntime"
        doc["max_abs_diff"] = float(max(np.nanmax(np.abs(g - w.numpy())) for g, w in zip(got, want, strict=True)))
    except ImportError:
        try:
            import onnx  # noqa: PLC0415

            onnx.checker.check_model(onnx.load(str(path)))
            doc["checked_with"] = "onnx.checker"
        except ImportError:
            doc["checked_with"] = "onnx_structure (no onnx or onnxruntime offline; parses the protobuf, runs nothing)"
            doc["structure"] = onnx_structure(path)
            doc["ok"] = doc["structure"]["nodes"] > 0 and doc["structure"]["outputs"] == output_names
    return doc


def export_all(out: Path, probe_tokens=None) -> dict[str, Any]:
    """ONNX of the combat net (policy and value heads), FightNet and V_run in `out`."""
    import torch  # noqa: PLC0415
    from combat_net import CombatNet, collate  # noqa: PLC0415
    from combat_tokens import VOCAB  # noqa: PLC0415

    out = Path(out)
    res: dict[str, Any] = {}

    class _Wrap(torch.nn.Module):
        def __init__(self, net) -> None:
            super().__init__()
            self.net = net

        def forward(self, types, ids, owners, feats, mask):
            return self.net(types, ids, owners, feats, mask)

    blob = torch.load(out / "combat_policy.pt", map_location="cpu", weights_only=True)
    net = CombatNet(VOCAB, blob["n_buckets"], d_model=blob.get("d_model", 128), layers=blob.get("layers", 2),
                    heads=blob.get("heads", 4))
    net.load_state_dict(blob["model"])
    if probe_tokens is None:
        from combat_tokens import N_FEATURES, TYPE_GLOBAL  # noqa: PLC0415

        types = torch.zeros(1, 6, dtype=torch.long)
        types[0, 0] = TYPE_GLOBAL
        batch = {"types": types, "ids": torch.zeros(1, 6, dtype=torch.long), "owners": torch.zeros(1, 6, dtype=torch.long),
                 "feats": torch.zeros(1, 6, N_FEATURES), "mask": torch.zeros(1, 6, dtype=torch.bool)}
    else:
        batch = collate([probe_tokens], torch.device("cpu"))
    names = ["types", "ids", "owners", "feats", "mask"]
    res["combat_net"] = export_onnx(_Wrap(net), tuple(batch[k] for k in names), names, ["value_logits", "policy_logits"],
                                    {**{k: {0: "batch", 1: "tokens"} for k in names},
                                     "value_logits": {0: "batch"}, "policy_logits": {0: "batch"}}, out / "combat_net.onnx")
    pt = out / "run_value.pt"
    if pt.is_file():
        from run_train import models  # noqa: PLC0415

        fight, value = models.FightNet(), models.RunValueNet()
        models.load_torch(pt, fight, value)
        counts = models.E_SIZE + models.F_SIZE + models.G_SIZE
        res["fight_net"] = export_onnx(fight, (torch.zeros(1, models.FIGHT_DENSE), torch.zeros(1, counts)), ["dense", "counts"],
                                       ["logits"], {"dense": {0: "batch"}, "counts": {0: "batch"}, "logits": {0: "batch"}},
                                       out / "fight_net.onnx")
        res["run_value"] = export_onnx(value, (torch.zeros(1, models.VALUE_DENSE), torch.zeros(1, counts)), ["dense", "counts"],
                                       ["floor_logits", "end_hp"],
                                       {"dense": {0: "batch"}, "counts": {0: "batch"}, "floor_logits": {0: "batch"},
                                        "end_hp": {0: "batch"}}, out / "run_value.onnx")
    return res


def train_generation(gen: int, shards_root: Path, agent_root: Path, *, device: str | None = None, epochs: int = 16,
                     max_fights: int = MAX_FIGHTS, threads: int = 6, vrun: bool = True, export: bool = True) -> Path:
    """Train generation `gen`'s candidate (the next generation's) from the last GEN_WINDOW generations' shards."""
    from run_train import shards as rs  # noqa: PLC0415

    cand = Path(agent_root) / f"gen{gen + 1}" / "candidate"
    cand.mkdir(parents=True, exist_ok=True)
    mans = window_manifests(shards_root, gen)
    if not mans:
        raise ValueError(f"no shards of generations {max(0, gen - GEN_WINDOW + 1)}..{gen} under {Path(shards_root).name}")
    t0 = time.perf_counter()
    data = build_training_set(mans, max_fights=max_fights, seed=gen)
    doc: dict[str, Any] = {
        "generation_trained_from": gen, "candidate_generation": gen + 1, "window": GEN_WINDOW,
        "blocks": len(mans), "runs": sum(int(m["runs"]) for _, m in mans),
        "runs_by_generation": {str(g): sum(int(m["runs"]) for _, m in mans if int(m["generation"]) == g)
                               for g in sorted({int(m["generation"]) for _, m in mans})},
        "samples_available": {str(int(g)): int(n) for g, n in data["_samples_by_generation"]},
        "fights_available": int(data["_fights_available"][0]), "fights_kept": int(data["_fights_kept"][0]),
        "samples_trained": int(data["lengths"].size), "target": "root visits (live search has no Gumbel root)",
        "validation": "by run (train_combat splits on the `fights` column, which holds the run index)",
        "build_s": round(time.perf_counter() - t0, 1)}
    doc["combat"] = train_combat_prior(data, cand, device=device, epochs=epochs, threads=threads, seed=gen)
    write_prior_model_json(cand, gen + 1)
    base_sims = int(mans[0][1]["base_sims"])
    if vrun:
        paths = [d / n for d, m in mans for n in m.get("run_shards", [])]
        doc["run_value"] = train_run_value(paths, cand, generation=gen, base_sims=base_sims,
                                           ascension=int(mans[0][1]["ascension"]), threads=threads)
    del rs
    if export:
        doc["onnx"] = export_all(cand)
    doc["cpu_min_total"] = round(doc["combat"]["cpu_min"] + doc.get("run_value", {}).get("cpu_min", 0.0), 2)
    doc["wall_min_total"] = round(doc["combat"]["wall_min"] + doc.get("run_value", {}).get("wall_min", 0.0), 2)
    # V_run drives macro choices only once its DESIGN 5.3 keep conditions are shown; until then the rules play outside fights
    write_json(cand / "agent.json", {"generation": gen + 1, "macro": "rules", "trained_from": gen, "written": utc_now(),
                                     "note": "macro stays rules until V_run passes DESIGN 5.3 (Brier vs base rate and V0, ECE)"})
    write_json(cand / "train.json", doc)
    shutil_rmtree(cand / "work")
    return cand


def shutil_rmtree(p: Path) -> None:
    import shutil  # noqa: PLC0415

    shutil.rmtree(p, ignore_errors=True)


# ------------------------------------------------------------------ lab markers


def ready_markers(train_dir: Path) -> list[tuple[int, Path]]:
    """`LAB/train/ready-<gen>` markers, lowest generation first."""
    out = []
    d = Path(train_dir)
    for p in d.glob("ready-*") if d.is_dir() else []:
        m = re.fullmatch(r"ready-(\d+)", p.name)
        if m:
            out.append((int(m.group(1)), p))
    return sorted(out)


def mark_ready(lab: Path, gen: int, shards_root: Path, budget: int = BUDGET_RUNS) -> Path | None:
    """Write `LAB/train/ready-<gen>` once the generation's self-play budget is spent; never twice (ready- or done-)."""
    d = Path(lab) / "train"
    if (d / f"ready-{gen}").exists() or (d / f"done-{gen}").exists():
        return None
    if runs_in_generation(shards_root, gen) < budget:
        return None
    d.mkdir(parents=True, exist_ok=True)
    p = d / f"ready-{gen}"
    p.write_text(utc_now() + "\n", encoding="utf-8")
    return p


# ------------------------------------------------------------------ promotion maths


def boss_kill(row: dict) -> bool:
    """Act-1 boss killed: the run won, or ended past the Act 1 boss floor."""
    return bool(row.get("win")) or int(row.get("floor") or 0) > ACT1_LAST_FLOOR


def cluster_bootstrap_lower(diffs: list[float], alpha: float = ALPHA, b: int = BOOT_B, seed: int = 20261023) -> dict[str, float]:
    """One-sided (1 - alpha) lower bound of the mean of paired differences, resampling whole seeds (clusters = pairs).
    `p` is the share of resampled means at or below zero (with the +1 correction)."""
    x = np.asarray(diffs, dtype=np.float64)
    n = x.size
    if n == 0:
        return {"n": 0, "mean": float("nan"), "lower": float("nan"), "p": 1.0, "boot": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(b)
    step = max(1, 4_000_000 // n)
    for s in range(0, b, step):
        e = min(b, s + step)
        means[s:e] = x[rng.integers(0, n, size=(e - s, n))].mean(axis=1)
    return {"n": int(n), "mean": float(x.mean()), "lower": float(np.quantile(means, alpha)),
            "p": float((np.sum(means <= 0) + 1) / (b + 1)), "boot": int(b)}


def mcnemar_exact_one_sided(b: int, c: int) -> float:
    """Exact one-sided McNemar: b = pairs the candidate won and the base did not, c = the reverse.
    P(X >= b), X ~ Binomial(b + c, 1/2). No discordant pair gives 1."""
    n = b + c
    if n == 0:
        return 1.0
    return sum(math.comb(n, k) for k in range(b, n + 1)) / 2 ** n


def paired_binary(cand: dict[str, bool], base: dict[str, bool]) -> dict[str, Any]:
    seeds = sorted(set(cand) & set(base))
    b = sum(1 for s in seeds if cand[s] and not base[s])
    c = sum(1 for s in seeds if base[s] and not cand[s])
    return {"pairs": len(seeds), "cand": sum(cand[s] for s in seeds), "base": sum(base[s] for s in seeds), "b": b, "c": c,
            "p": mcnemar_exact_one_sided(b, c)}


def decide(floor: dict, boss: dict, wins: dict, cand_win_rate: float, base_win_rate: float, alpha: float = ALPHA) -> dict[str, Any]:
    """Promote only if every co-primary passes: the paired floor's lower bound above 0, the Act-1 boss-kill McNemar below
    alpha with the candidate ahead, and, once both arms win at least 10% of runs, the wins McNemar likewise."""
    checks = {"floor": bool(floor["lower"] > 0), "boss_kill": bool(boss["p"] < alpha and boss["b"] > boss["c"])}
    wins_active = cand_win_rate >= WIN_GATE and base_win_rate >= WIN_GATE
    if wins_active:
        checks["wins"] = bool(wins["p"] < alpha and wins["b"] > wins["c"])
    return {"checks": checks, "wins_co_primary": wins_active, "promote": all(checks.values()), "alpha": alpha}


def arm_rows(arm_dir: Path) -> list[dict]:
    return [json.loads(x) for x in (Path(arm_dir) / "runs.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def arm_summary(rows: list[dict]) -> dict[str, Any]:
    import gen0  # noqa: PLC0415

    t = gen0.engine_block(rows)
    k = sum(boss_kill(r) for r in rows)
    lo, hi = gen0.wilson(k, len(rows))
    return {**t, "boss_kills": k, "boss_kill_rate": k / len(rows) if rows else None, "boss_kill_lo": lo, "boss_kill_hi": hi}


def compare_arms(cand_rows: list[dict], base_rows: list[dict], alpha: float = ALPHA, boot: int = BOOT_B) -> dict[str, Any]:
    """The promotion table's numbers from two arms' rows, paired on seed."""
    c = {r["seed"]: r for r in cand_rows}
    b = {r["seed"]: r for r in base_rows}
    seeds = sorted(set(c) & set(b))
    diffs = [float(c[s]["floor"]) - float(b[s]["floor"]) for s in seeds]
    floor = cluster_bootstrap_lower(diffs, alpha, boot)
    boss = paired_binary({s: boss_kill(c[s]) for s in seeds}, {s: boss_kill(b[s]) for s in seeds})
    wins = paired_binary({s: bool(c[s]["win"]) for s in seeds}, {s: bool(b[s]["win"]) for s in seeds})
    cw = sum(bool(c[s]["win"]) for s in seeds) / max(1, len(seeds))
    bw = sum(bool(b[s]["win"]) for s in seeds) / max(1, len(seeds))
    return {"paired_seeds": len(seeds), "floor": floor, "boss_kill": boss, "wins": wins,
            "decision": decide(floor, boss, wins, cw, bw, alpha)}


# ------------------------------------------------------------------ the E-block and the promotion run


def check_disjoint(seeds: list[str], others: dict[str, list[str]]) -> None:
    """Raise if the block shares a seed with any of the named seed collections."""
    mine = set(seeds)
    for name, other in others.items():
        both = mine & set(other)
        if both:
            raise ValueError(f"{len(both)} seeds also in {name}")


def known_seed_sets(seed_dir: Path, lab_runs: Path | list[Path] | None, e_name: str) -> dict[str, list[str]]:
    """Every committed seed block except this one, and the seeds the lab's self-play blocks actually played."""
    import gen0  # noqa: PLC0415

    out: dict[str, list[str]] = {}
    for f in sorted(Path(seed_dir).glob("*.json")):
        if f.stem != e_name:
            out[f.stem] = gen0.load_seeds(f)
    sp: list[str] = []
    for d in ([lab_runs] if isinstance(lab_runs, (str, Path)) else list(lab_runs or [])):
        for f in Path(d).glob("*/runs.jsonl"):
            with contextlib.suppress(OSError, ValueError):
                sp += [json.loads(x)["seed"] for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    if lab_runs:
        out["self-play runs"] = sp
    return out


def run_arm(seeds: list[str], arm_dir: Path, agent: Agent, config: str, procs: int, ascension: int = 10) -> list[dict]:
    """Play the seeds with `agent`, or reuse the arm if the same seeds and config were already played there."""
    import gen0  # noqa: PLC0415

    arm_dir = Path(arm_dir)
    s = arm_dir / "summary.json"
    if s.is_file():
        old = read_json(s)
        rows = arm_rows(arm_dir)
        if old.get("config") == config and old.get("agent") == agent.describe() and {r["seed"] for r in rows} == set(seeds):
            return rows
    gen0.run_block(seeds, config, procs, arm_dir, ascension=ascension, agent=agent)
    return arm_rows(arm_dir)


def promote(cand: Path, base: Path | None, seeds_file: Path, *, agent_root: Path, procs: int = 16, config: str = "a3-v1",
            ascension: int = 10, lab: Path | None = None, lab_runs: list[Path] | None = None, boot: int = BOOT_B) -> dict[str, Any]:
    """Play candidate and base on the same E-block seeds, test, and record the decision."""
    import gen0  # noqa: PLC0415

    seeds = gen0.load_seeds(seeds_file)
    name = Path(seeds_file).stem
    if name.startswith("SP-"):
        raise ValueError("an E-block is never named like a self-play stream")
    check_disjoint(seeds, known_seed_sets(Path(seeds_file).parent, lab_runs, name))
    cand_agent = load_agent(cand)
    base_agent = load_agent(base, agent_root=None if base else agent_root)
    arm = f"{name}-{config}"
    cand_rows = run_arm(seeds, Path(cand) / "arms" / arm, cand_agent, config, procs, ascension)
    base_dir = Path(base_agent.path) if base_agent.path else Path(agent_root) / "gen0"
    base_rows = run_arm(seeds, base_dir / "arms" / arm, base_agent, config, procs, ascension)
    res = compare_arms(cand_rows, base_rows, ALPHA, boot)
    doc = {"written": utc_now(), "seed_block": name, "n_seeds": len(seeds), "config": config, "procs": procs,
           "alpha": ALPHA, "cand": {"agent": cand_agent.describe(), "summary": arm_summary(cand_rows)},
           "base": {"agent": base_agent.describe(), "summary": arm_summary(base_rows)}, **res,
           "decision_text": "promoted" if res["decision"]["promote"] else "rejected"}
    write_json(Path(cand) / "promotion.json", doc)
    f, bk = doc["floor"], doc["boss_kill"]
    line = "\t".join(str(x) for x in (doc["written"], cand_agent.generation, base_agent.generation, name, doc["n_seeds"],
                                      f"{f['mean']:.4f}", f"{f['lower']:.4f}", f"{f['p']:.5f}", bk["cand"], bk["base"],
                                      f"{bk['p']:.5f}", doc["cand"]["summary"]["wins"], doc["base"]["summary"]["wins"],
                                      doc["decision_text"]))
    tsv = Path(agent_root) / "promotions.tsv"
    if not tsv.exists():
        tsv.write_text("written\tcand_gen\tbase_gen\tseed_block\tseeds\td_floor\tfloor_lower\tp_floor\tboss_cand\tboss_base\t"
                       "p_boss\twins_cand\twins_base\tdecision\n", encoding="utf-8")
    with tsv.open("a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    if res["decision"]["promote"]:
        write_json(Path(agent_root) / "current.json", {"generation": cand_agent.generation, "dir": str(Path(cand).resolve()),
                                                        "promoted": doc["written"], "seed_block": name})
        if lab is not None:
            p = Path(lab) / "score" / "pending"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(utc_now() + "\n", encoding="utf-8")
    return doc


def resolve_e_block(template: str, k: int, n: int = 2000) -> Path:
    """The E-block for candidate generation `k`: `template` with {k} for the generation and {date} for today (UTC).
    A file `E<k>-*.json` that already exists is used as it is (E1-2026-10-23 is committed); otherwise the block is
    created by the seed rule, under a name no other generation has, so each promotion meets a fresh block."""
    import gen0  # noqa: PLC0415

    path = Path(template.format(k=k, date=datetime.now(UTC).strftime("%Y-%m-%d")))
    existing = sorted(path.parent.glob(f"E{k}-*.json"))
    if existing:
        return existing[0]
    gen0.write_seed_file(path.stem, n, path)
    return path


def run_promote_subprocess(emu_python: str, cand: Path, base: Path | None, seeds_file: Path, agent_root: Path, procs: int,
                           lib: str | None, lab: Path | None) -> int:
    """From the train job (GPU venv): promotion plays emulators, which run in the emulator venv."""
    cmd = [emu_python, str(ROOT / "rl" / "gen0.py"), "promote", "--cand", str(cand), "--seeds-file", str(seeds_file),
           "--agent-root", str(agent_root), "--procs", str(procs)]
    if base:
        cmd += ["--base", str(base)]
    if lib:
        cmd += ["--lib", lib]
    if lab:
        cmd += ["--lab", str(lab)]
    return subprocess.run(cmd, check=False).returncode


__all__ = [n for n in dir() if not n.startswith("_")]
_ = dataclasses
