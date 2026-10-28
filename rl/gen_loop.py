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

The live search has no Gumbel root (A3 report: no Gumbel knobs exist), so the target is the root visit distribution by
default; `--policy-target completed_q` trains on the softmax of the root's completed `qvals` instead.

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
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, field
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
REVIEW_GUARD_FLAG = "SPIRELINE_REVIEW_GUARD"  # tools/lab/promotion_guard.FLAG: the review's guard decides when set; off by default
REVIEW_GUARD_ON_VALUES = ("1", "on", "true", "yes")  # tools/lab/promotion_guard.ON_VALUES (a test keeps them equal)
# D33 (2026-10-25): the paired rules-against-vrun A/B (VRUN_SWEEP.md 1.5) was decided, so the value-head macro is the macro of every new
# candidate whose V_run is kept, with no environment variable. SPIRELINE_MACRO_VRUN stays as an explicit opt-out: "rules" (or off, 0,
# false, no) makes `train_generation` write macro rules; anything else, or unset, leaves the default.
MACRO_VRUN_FLAG = "SPIRELINE_MACRO_VRUN"
# A candidate written under a generation number that skips ones (`ready-<gen>-g<target>`) says so in train.json and agent.json (D33 addendum (b)).
LINEAGE_DECISION = "D33 addendum (b)"
LINEAGE_DIFFERENCE = "macro {macro} with {fit} (D33, 2026-10-25); generation 4's two attempts were rules-macro candidates"
LINEAGE_FIT_TEXT = {"seed": "the A/B's V_run fit", "new": "its own new V_run fit", "none": "no V_run fit"}
MACRO_OPT_OUT_VALUES = ("rules", "off", "0", "false", "no")
MACRO_SWITCH_NOTE = ("D33 (2026-10-25): the macro switch is on by default after the paired rules-against-vrun A/B "
                     "(VRUN_SWEEP.md 1.5); SPIRELINE_MACRO_VRUN=rules opts a candidate out")
# which prior produced a run's decisions: the row, the record and the npz part (an int8 per run) all carry it
# "uniform": at least one call was answered by the uniform prior of the memory floor (prior_client), whatever else answered
PRIOR_PATHS = ("none", "cpu", "server", "mixed", "uniform")
PRIOR_PATH_CODES = {"none": 0, "cpu": 1, "server": 2, "mixed": 3, "uniform": 4}  # -1: unknown (a part written before the field)
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


# The run mask version (native.RUN_MASK_VERSIONS) a candidate written by `train_generation` carries in its agent.json;
# an agent.json without the key is version 1 (generations 0-3 and every candidate saved before the key existed).
CANDIDATE_MASK_VERSION = 4
# The versions `load_agent` accepts (native.RUN_MASK_VERSIONS; not imported here, the library loads lazily): 1 is the mask
# as it was, 2 adds the discard actions (D32), 3 the AnyTime potions' use actions (D34) on the closed-belt screens, 4 the same mask with Welcome
# to Wongos' Leave shown to the policies (D34; generations 0-7 and every candidate saved before keep the version they carry).
AGENT_MASK_VERSIONS = (1, 2, 3, 4)


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
    # the transport of the combat prior, not part of the agent's identity (`describe` leaves them out): the batched prior
    # server's endpoint and this job's key (docs/rl/GPU_INFERENCE.md). Set by `gen0.run_block` after it started the server.
    prior_server: str | None = None
    prior_authkey: bytes | None = field(default=None, repr=False)
    # the version of the run action mask this agent's runs are built with (agent.json `mask_version`, absent: 1); it only
    # changes what the mask carries, never what the agent chooses (run_train.engine._legal hides the potion actions)
    mask_version: int = 1

    def search_extra(self) -> dict[str, Any]:
        """`live_search.search_live` keyword arguments that switch the tree to PUCT with the prior; {} for none.
        With `prior_server` the champion's in-process prior is replaced by a `prior_client.RemotePrior` to that server."""
        if not self.prior_dir:
            return {}
        extra = dict(_champion_settings(self.prior_dir))
        if self.prior_server:
            if self.prior_authkey is None:
                raise ValueError("prior_server needs prior_authkey")
            import prior_client  # noqa: PLC0415 -- numpy only, but absent from builds before the server

            champion = _champion(self.prior_dir)
            extra["prior_fn"] = prior_client.RemotePrior(policy=str(champion.policy), probe=champion.probe,
                                                         endpoint=self.prior_server, authkey=self.prior_authkey,
                                                         policy_sha256=champion.policy_sha256)
        return extra

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
def _champion(prior_dir: str):
    import combat_prior  # noqa: PLC0415 -- native

    champion = combat_prior.find(prior_dir)
    if champion is None:
        raise FileNotFoundError(f"no combat prior (model.json and combat_policy.pt) in {Path(prior_dir).name}")
    return champion


@lru_cache(maxsize=8)
def _champion_settings(prior_dir: str) -> tuple[tuple[str, Any], ...]:
    return tuple(_champion(prior_dir).live_settings().items())


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
    mask_version = meta.get("mask_version", 1)
    if isinstance(mask_version, bool) or mask_version not in AGENT_MASK_VERSIONS:
        raise ValueError(f"{p.name}/agent.json: mask_version {mask_version!r} is not one of {list(AGENT_MASK_VERSIONS)}")
    return Agent(str(p), generation, prior, mode, str(npz) if mode == "vrun" else None,
                 float(meta.get("tau", 0.0)), float(meta.get("eps", 0.0)), mask_version=int(mask_version))


def active_agent_extra() -> dict[str, Any]:
    """`search_configs.SearchConfig.search_kwargs` adds this: the prior of the agent named by SPIRELINE_GEN_DIR (score-real's
    lockstep player builds its search from a config alone), nothing when the variable is unset."""
    d = os.environ.get(ENV_GEN_DIR)
    if not d:
        return {}
    # built from a directory, so it has no server; the lockstep and the score never receive a RemotePrior (a test pins it)
    return dataclasses.replace(load_agent(d), prior_server=None, prior_authkey=None).search_extra()


def replay_note(prior_path: str | None, what: str) -> bool:
    """Log one line (never fail) when a run or sample that is being replayed was not played on the CPU path: a replay always
    uses the in-process prior, so a `server`, `mixed` or `uniform` run may not be reproduced decision for decision. True when noted."""
    if prior_path in (None, "cpu", "none"):
        return False
    print(f"replay: {what} was played with prior_path {prior_path!r}; the CPU replay may not be exact", file=sys.stderr, flush=True)
    return True


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

    def finish(self, session, moves: int, won: bool, steps_cap_hit: bool = False, prior_path: str | None = None,
               prior_calls: dict | None = None) -> dict[str, Any]:
        """The run's RunRecord. `prior_path` ("none", "cpu", "server", "mixed" or "uniform") and `prior_calls` say which prior played it."""
        from run_train import FEATURE_VERSION, shards  # noqa: PLC0415
        from run_train.vocab import WON_FLOOR  # noqa: PLC0415

        self.run.moves = moves
        info, stats = session.info, session.stats
        record = {
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
        if prior_path is not None:
            record["prior_path"] = prior_path
        if prior_calls is not None:
            record["prior_calls"] = dict(prior_calls)
        return record


# ------------------------------------------------------------------ shards of a self-play block


def prior_path_code(tag: str | None) -> int:
    return PRIOR_PATH_CODES.get(tag, -1) if tag is not None else -1


def write_part(path: Path, runs: list[tuple[str, dict | None]], generation: int, prior_paths: list[str | None] | None = None) -> int:
    """Write the combat samples of `runs` (seed, arrays-or-None) as one npz in `prior_loop`'s layout; return the sample count.
    `fights` is the run's index in the part (so a split by that column is a split by run). With `prior_paths` (one tag per run)
    the part also holds `prior_path`, an int8 per run aligned with `seeds` (0 none, 1 cpu, 2 server, 3 mixed, 4 uniform, -1 unknown)."""
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
    if prior_paths is not None:
        if len(prior_paths) != len(runs):
            raise ValueError("one prior_path per run")
        out["prior_path"] = np.array([prior_path_code(t) for t in prior_paths], dtype=np.int8)
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


def shard_run_seeds(directory: Path, generation: int) -> set[str]:
    """The seeds of `generation`'s run records in the run shards of `directory`."""
    from run_train import shards  # noqa: PLC0415

    return {str(r["seed"]) for p in shards.shard_paths(Path(directory)) for r in shards.read_shard(p)
            if r.get("generation") == generation}


_PRIOR_TAGS = {code: tag for tag, code in PRIOR_PATH_CODES.items()}


class ShardSink:
    """The main process's writer for one self-play block: run shards (RunRecords) and combat parts, and `shards.json`,
    which names the generation that played them. The manifest is rewritten at every flush, so an interrupted block is
    still readable.

    The writer is idempotent on a resume. The lab resumes an interrupted job in the same directory and the job replays its
    seed stream from the first seed, so a sink that finds the block's run shards holds those seeds as `done`: `add` ignores
    a record of a `done` seed, `gen0.run_block` does not play it, the combat parts that cover them are kept and the new
    parts are numbered on from them, and `runs`, `samples` and `prior_paths` start from what is on disk. A flush writes the
    combat part, then the run shard, then the manifest, so a kill leaves either nothing of a part's runs in the shards
    (they are replayed) or the part beside them (it is adopted here)."""

    def __init__(self, out_dir: Path, generation: int, *, config: str, base_sims: int, ascension: int, agent: dict,
                 part_runs: int = PART_RUNS) -> None:
        from run_train import shards  # noqa: PLC0415

        self.dir = Path(out_dir) / "shards"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.generation, self.part_runs = generation, part_runs
        self.meta = {"kind": "selfplay-shards", "generation": generation, "config": config, "base_sims": base_sims,
                     "ascension": ascension, "agent": agent}
        self.writer = shards.ShardWriter(self.dir, 1 << 30)  # only `flush` writes a run shard, after the combat part
        self.pending: list[tuple[str, dict | None]] = []
        self.pending_paths: list[str | None] = []
        self.parts: list[str] = []
        self.runs = self.samples = 0
        self.prior_paths: dict[str, int] = {}  # runs by the prior that played them ("unknown": a record without the field)
        self.done: set[str] = set()  # the seeds this block holds: written to a run shard, or pending
        self.duplicates_skipped = 0  # records `add` ignored because their seed was `done`
        self._resume()
        self.resumed_runs = self.runs

    def _resume(self) -> None:
        """Take over what an interrupted block of this generation left in the directory (nothing on disk is rewritten)."""
        self.done = shard_run_seeds(self.dir, self.generation)
        covered: set[str] = set()
        while self.done:
            path = self.dir / f"combat-g{self.generation}-{len(self.parts):04d}.npz"
            try:
                blob = read_part(path)
            except (OSError, ValueError, EOFError, zipfile.BadZipFile):
                break
            seeds = [str(s) for s in blob["seeds"]]
            if (int(blob["generation"][0]) != self.generation or not set(seeds) <= self.done or covered & set(seeds)
                    or len(set(seeds)) != len(seeds)):
                break  # a stale part (its runs are not in the shards, or another part holds them) is overwritten, not adopted
            covered.update(seeds)
            self.parts.append(path.name)
            self.samples += int(blob["lengths"].size) if "lengths" in blob else 0
            codes = blob["prior_path"].tolist() if "prior_path" in blob else [-1] * len(seeds)
            for code in codes:
                tag = _PRIOR_TAGS.get(int(code), "unknown")
                self.prior_paths[tag] = self.prior_paths.get(tag, 0) + 1
        self.runs = len(self.done)
        if self.runs > len(covered):  # run records whose combat samples were not kept (a block from before this writer)
            self.prior_paths["unknown"] = self.prior_paths.get("unknown", 0) + self.runs - len(covered)

    def add(self, record: dict, arrays: dict | None) -> None:
        if record["generation"] != self.generation:
            raise ValueError(f"a shard is tagged with the generation that played it: {record['generation']} in a "
                             f"generation {self.generation} block")
        if str(record["seed"]) in self.done:  # a replayed seed: its run record and combat part are already held
            self.duplicates_skipped += 1
            return
        self.done.add(str(record["seed"]))
        self.writer.add(record)
        self.pending.append((record["seed"], arrays))
        tag = record.get("prior_path")
        self.pending_paths.append(tag)
        self.prior_paths[tag or "unknown"] = self.prior_paths.get(tag or "unknown", 0) + 1
        self.runs += 1
        if len(self.pending) >= self.part_runs:
            self.flush()

    def flush(self) -> None:
        if self.pending:
            name = f"combat-g{self.generation}-{len(self.parts):04d}.npz"
            self.samples += write_part(self.dir / name, self.pending, self.generation, self.pending_paths)
            self.parts.append(name)
            self.pending, self.pending_paths = [], []
        self.writer.flush()
        self._manifest()

    def _manifest(self) -> None:
        from run_train import shards  # noqa: PLC0415

        write_json(self.dir / "shards.json", {**self.meta, "runs": self.runs, "samples": self.samples,
                                              "resumed_runs": self.resumed_runs, "duplicates_skipped": self.duplicates_skipped,
                                              "prior_paths": dict(sorted(self.prior_paths.items())),
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


def build_training_set(manifests: list[tuple[Path, dict]], *, max_fights: int = MAX_FIGHTS, seed: int = 0,
                       exclude_prior_paths: tuple[str, ...] = ()) -> dict[str, Any]:
    """Merge the combat parts of the window with `prior_loop.loop.merge_datasets` and sample at most `max_fights` fights.
    Returns the arrays plus counts per generation. `exclude_prior_paths` (tags of `PRIOR_PATHS`) drops the runs a part's
    `prior_path` codes tag so; a part from before the field has none and is never dropped.

    A (generation, seed) counts once: a block that was resumed before `ShardSink` was idempotent holds its replayed runs again,
    so the first part to name a seed keeps its rows and the later copies' rows are dropped (`_runs_dropped`, `_runs_distinct`
    and `_samples_dedup_dropped` say how many)."""
    from prior_loop.loop import merge_datasets  # noqa: PLC0415

    bad = [t for t in exclude_prior_paths if t not in PRIOR_PATH_CODES]
    if bad:
        raise ValueError(f"unknown prior path {bad[0]!r}; one of {', '.join(PRIOR_PATHS)}")
    excluded = np.array([PRIOR_PATH_CODES[t] for t in exclude_prior_paths], dtype=np.int8)
    sources, per_gen, ixs, qs, keeps, firsts = [], {}, [], [], [], []
    seen: set[tuple[int, str]] = set()
    runs_dropped = 0
    for d, m in manifests:
        for name in m.get("combat_parts", []):
            blob = read_part(d / name)
            g = int(m["generation"])
            if int(blob["generation"][0]) != g:  # the tag is checked where the data is read, not only where it is written
                raise ValueError(f"{name} is tagged generation {int(blob['generation'][0])} but its manifest says {g}")
            if "lengths" not in blob:  # a part whose runs recorded no searched decision
                continue
            keep = np.ones(blob["lengths"].size, dtype=bool)
            if excluded.size and "prior_path" in blob:  # `fights` is the run's index in the part, the index of its code
                keep = ~np.isin(blob["prior_path"][blob["fights"]], excluded)
            first = np.ones(len(blob["seeds"]), dtype=bool)  # per run in the part: its seed is not an earlier part's
            for i, s in enumerate(blob["seeds"]):
                if (g, str(s)) in seen:
                    first[i] = False
                seen.add((g, str(s)))
            runs_dropped += int((~first).sum())
            fresh = first[blob["fights"]]  # `fights` is the run's index in the part
            keeps.append(keep)
            firsts.append(fresh)
            per_gen[g] = per_gen.get(g, 0) + int((keep & fresh).sum())
            ixs.append(blob["fight_ix"])
            qs.append(blob.get("qvals"))
            sources.append((len(sources), d / name))
    if not sources:
        raise ValueError("no combat samples in the window")
    merged = merge_datasets(sources)  # offsets each part's run index, so a run id is unique over the window
    if all(q is not None for q in qs):  # merge_datasets keeps prior_loop's keys; the completed-Q target needs these
        merged["qvals"] = np.concatenate(qs)
    ix_all = np.concatenate(ixs)
    prior_keep, fresh_all = np.concatenate(keeps), np.concatenate(firsts)
    keep_all = prior_keep & fresh_all
    n_excluded = int((~prior_keep & fresh_all).sum())
    if not keep_all.all():
        if not keep_all.any():
            raise ValueError("no combat samples in the window after excluding prior paths " + ",".join(exclude_prior_paths))
        merged = _take_rows(merged, keep_all)
        ix_all = ix_all[keep_all]
    key = merged["fights"] * 4096 + ix_all  # a fight = (run, fight_ix in the run)
    uniq, inv = np.unique(key, return_inverse=True)
    n_all = int(uniq.size)
    if n_all > max_fights:
        pick = np.random.default_rng(seed).permutation(n_all)[:max_fights]
        keep = np.isin(inv, pick)
        merged = _take_rows(merged, keep)
        ix_all = ix_all[keep]
        kept = int(np.unique(key[keep]).size)
    else:
        kept = n_all
    merged["_samples_excluded"] = np.array([n_excluded])
    merged["_runs_dropped"] = np.array([runs_dropped])
    merged["_runs_distinct"] = np.array([len(seen)])
    merged["_samples_dedup_dropped"] = np.array([int((~fresh_all).sum())])
    merged["_fights_available"] = np.array([n_all])
    merged["_fights_kept"] = np.array([kept])
    merged["_fight_ix"] = ix_all  # each row's fight index within its run (an underscore key: saved sets do not carry it)
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
                       layers: int = 2, threads: int = 6, seed: int = 0, policy_target: str = "visits",
                       q_temperature: float | None = None, epoch_end: Any = None) -> dict[str, Any]:
    """Fit the policy and value heads with `train_combat.train` (prior_loop's trainer, its settings and its checkpoints).
    `epoch_end(epoch, stats)` is `train_combat.train`'s hook (the prior sweep's stop file); None leaves the call as it was."""
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
                              device=dev, policy_target=policy_target,
                              q_temperature=train_combat.Q_TEMPERATURE if q_temperature is None else q_temperature)
    saved = {n: getattr(train_combat, n) for n in ("DATASET", "CHECKPOINT", "POLICY_CHECKPOINT")}
    cap = io.StringIO()
    t0, c0 = time.perf_counter(), time.process_time()
    try:
        train_combat.DATASET, train_combat.CHECKPOINT = npz, out / "combat_net.pt"
        train_combat.POLICY_CHECKPOINT = out / "combat_policy.pt"
        with contextlib.redirect_stdout(cap):
            if epoch_end is None:
                train_combat.train(args)
            else:
                train_combat.train(args, epoch_end=epoch_end)
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
            "samples": int(data["lengths"].size), "policy_target": policy_target,
            "q_temperature": args.q_temperature if policy_target == "completed_q" else None}


def write_prior_model_json(out: Path, generation: int, probe: bool = False) -> dict:
    import combat_prior  # noqa: PLC0415

    sha = combat_prior._sha256_file(Path(out) / "combat_policy.pt")  # noqa: SLF001
    doc = {"kind": combat_prior.KIND, "iteration": generation, "vocab_sha256": combat_prior.vocab_sha256(), "policy_sha256": sha,
           "probe": probe, "search": dict(PRIOR_SEARCH)}
    write_json(Path(out) / "model.json", doc)
    return doc


def train_run_value(shard_paths: list[Path], out: Path, *, generation: int, base_sims: int, ascension: int, threads: int = 6,
                    device: str | None = None) -> dict:
    """Fit FightNet and V_run (DMC target, `run_train.train.fit`) on the window's run records, then the 5-fold out-of-fold fits
    the keep verdict rests on (D33, `keep.cross_fit_keep`). `device` (the train job's) puts the fits there; none: the CPU."""
    import torch  # noqa: PLC0415
    from run_train import keep  # noqa: PLC0415
    from run_train import train as rt  # noqa: PLC0415
    from run_train.engine import CombatBudget  # noqa: PLC0415

    t0, c0 = time.perf_counter(), time.process_time()
    pt = Path(out) / "run_value.pt"
    rt.fit(shard_paths, None, pt, generation=generation, combat_budget=CombatBudget(base_sims, base_sims), ascension=ascension,
           config=rt.TrainConfig(value_generations=GEN_WINDOW, threads=threads, **({"device": device} if device else {})),
           evaluate=keep.evaluate_value_net, cross_fit=keep.cross_fit_keep)
    meta = torch.load(pt, map_location="cpu", weights_only=True)["meta"]
    value = dict(meta.get("value") or {})
    return {"wall_min": (time.perf_counter() - t0) / 60, "cpu_min": (time.process_time() - c0) / 60,
            "fight": meta.get("fight"), "value": value, "config": meta.get("config")}


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


def vrun_keep_passed(keep_doc: dict[str, Any] | None, cand: Path) -> bool:
    """Whether the V_run keep record passes every computable condition and the candidate holds its export."""
    return bool(keep_doc and keep_doc.get("all_computable_pass")) and (Path(cand) / "run_value.npz").is_file()


def macro_vrun_on(flag: bool | None = None) -> bool:
    """Whether the macro switch is on: `flag` when given, else on unless SPIRELINE_MACRO_VRUN opts out (`MACRO_OPT_OUT_VALUES`)."""
    if flag is not None:
        return bool(flag)
    return os.environ.get(MACRO_VRUN_FLAG, "").strip().lower() not in MACRO_OPT_OUT_VALUES


def macro_for(keep_doc: dict[str, Any] | None, cand: Path, enabled: bool | None = None) -> str:
    """`vrun` when the keep record passes (`vrun_keep_passed`) and the macro switch is on (`macro_vrun_on(enabled)`), else `rules`.

    The switch is on by default since D33 (2026-10-25); SPIRELINE_MACRO_VRUN=rules is the opt-out. A candidate that plays
    V_run's macro is judged with its prior and V_run together; that is the decision (VRUN_SWEEP.md 1.5), and the E-block's
    rule (floor, boss-kill guard, F7 audit) is unchanged."""
    return "vrun" if vrun_keep_passed(keep_doc, cand) and macro_vrun_on(enabled) else "rules"


def file_sha256(path: Path | str) -> str:
    import hashlib  # noqa: PLC0415

    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def oof_numbers(keep_doc: dict[str, Any] | None) -> dict[str, Any] | None:
    """The out-of-fold numbers of a keep record's Act 1 head, for the manifest (none: the record has no judged head)."""
    head = ((keep_doc or {}).get("heads") or {}).get("act1_boss_kill") or {}
    if "brier_skill_vs_base_rate" not in head:
        return None
    return {"rows": head.get("rows"), "positive_runs": head.get("positive_runs"),
            "brier_skill_vs_base_rate": {k: head["brier_skill_vs_base_rate"].get(k) for k in ("value", "ci", "runs", "pass")},
            "brier_skill_vs_v0": {k: head["brier_skill_vs_v0"].get(k) for k in ("value", "ci", "pass")},
            "log_loss_pass": head["log_loss_vs_base_rate"]["pass"], "ece_pass": head["ece_bins_ge_30_positives"]["pass"]}


def carry_vrun_fit(cand: Path, keep_doc: dict[str, Any] | None, seed: Path | str | None) -> dict[str, Any]:
    """Which V_run fit the candidate carries (D33). A new fit that passes the keep test is the candidate's `run_value.npz`; one that
    fails is moved aside (`rejected-fit/`) and the seed fit (the A/B's, `train_generation(vrun_seed=)`) takes its place; with neither
    the candidate carries no kept fit. Returns the manifest entry: source (new / seed / none), kept, sha256 of the carried file, the
    out-of-fold numbers of the new fit, and the seed's path, sha256 and its sidecar record (`<stem>.keep.json`) when there is one."""
    import shutil  # noqa: PLC0415

    cand, npz = Path(cand), Path(cand) / "run_value.npz"
    passed = vrun_keep_passed(keep_doc, cand)
    seed_info = None
    if seed is not None and Path(seed).is_file():
        seed_info = {"path": Path(seed).as_posix(), "sha256": file_sha256(seed)}
        sidecar = Path(seed).with_name(Path(seed).stem + ".keep.json")
        if sidecar.is_file():
            seed_info["record"] = read_json(sidecar)
    entry: dict[str, Any] = {"rule": "a later fit replaces the seed fit only when it passes the keep test (D33)",
                             "new_fit_passed_keep": passed, "new_fit_oof": oof_numbers(keep_doc), "seed": seed_info}
    if passed:
        return {**entry, "source": "new", "kept": True, "sha256": file_sha256(npz)}
    if seed_info is not None:
        rejected = cand / "rejected-fit"
        rejected.mkdir(exist_ok=True)
        moved = {}
        for name in ("run_value.npz", "run_value.pt", "run_value.onnx"):
            if (cand / name).is_file():
                moved[name] = file_sha256(cand / name)
                shutil.move(str(cand / name), str(rejected / name))
        shutil.copyfile(seed, npz)
        return {**entry, "source": "seed", "kept": True, "sha256": file_sha256(npz), "rejected_fit": {"dir": rejected.name, "sha256": moved}}
    return {**entry, "source": "none", "kept": False, "sha256": file_sha256(npz) if npz.is_file() else None}


def lineage_note(gen: int, target: int, macro: str, fit_source: str) -> dict[str, Any] | None:
    """The lineage of a candidate written as generation `target` from generation `gen`'s shards: None for the ordinary gen + 1.
    `skips` are the generation numbers in between (their candidates were other policies and keep their own attempt counts)."""
    if target == gen + 1:
        return None
    return {"from_generation": gen, "skips": list(range(gen + 1, target)),
            "difference": LINEAGE_DIFFERENCE.format(macro=macro, fit=LINEAGE_FIT_TEXT.get(fit_source, LINEAGE_FIT_TEXT["none"])),
            "decision": LINEAGE_DECISION}


def train_generation(gen: int, shards_root: Path, agent_root: Path, *, device: str | None = None, epochs: int = 16,
                     max_fights: int = MAX_FIGHTS, threads: int = 6, vrun: bool = True, export: bool = True,
                     seed_dir: Path | None = None, retrain: int = 0, policy_target: str = "visits",
                     exclude_prior_paths: tuple[str, ...] = (), seed_dirs: Sequence[Path | str] = (),
                     vrun_seed: Path | str | None = None, candidate_generation: int | None = None) -> Path:
    """Train generation `gen`'s candidate (the next generation's, or `candidate_generation`'s) from the last GEN_WINDOW generations' shards. `seed_dirs`
    are further folders of E-blocks (the lab's generated ones) that no training seed may belong to. `vrun_seed` is a V_run fit
    (`run_value.npz`) the candidate carries unless its own new fit passes the keep test (`carry_vrun_fit`, D33). `candidate_generation`
    names the candidate's generation number when it is not gen + 1 (a new policy skipping the numbers between: it is written as
    gen<target>/candidate, its attempt count starts at zero, and train.json / agent.json carry a `lineage` note)."""
    from run_train import shards as rs  # noqa: PLC0415

    target = gen + 1 if candidate_generation is None else int(candidate_generation)
    if target <= gen:
        raise ValueError(f"a candidate trained from generation {gen} cannot be generation {target}")

    mans = window_manifests(shards_root, gen)
    if not mans:
        raise ValueError(f"no shards of generations {max(0, gen - GEN_WINDOW + 1)}..{gen} under {Path(shards_root).name}")
    window_seeds = [str(x) for d, m in mans for name in m.get("combat_parts", []) for x in read_part(d / name)["seeds"]]
    check_disjoint(window_seeds, known_seed_sets(seed_dir or COMMITTED_SEEDS, None, "", seed_dirs))  # no benchmark seed trains
    # a retrain after a rejection keeps the rejected candidate (and its promotion.json) where it is
    cand = Path(agent_root) / f"gen{target}" / ("candidate" if retrain == 0 else f"candidate-r{retrain}")
    cand.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    data = build_training_set(mans, max_fights=max_fights, seed=gen, exclude_prior_paths=exclude_prior_paths)
    doc: dict[str, Any] = {
        "generation_trained_from": gen, "candidate_generation": target, "window": GEN_WINDOW,
        "blocks": len(mans), "runs": sum(int(m["runs"]) for _, m in mans),
        "runs_by_generation": {str(g): sum(int(m["runs"]) for _, m in mans if int(m["generation"]) == g)
                               for g in sorted({int(m["generation"]) for _, m in mans})},
        "samples_available": {str(int(g)): int(n) for g, n in data["_samples_by_generation"]},
        "fights_available": int(data["_fights_available"][0]), "fights_kept": int(data["_fights_kept"][0]),
        **({"excluded_prior_paths": list(exclude_prior_paths), "samples_excluded": int(data["_samples_excluded"][0])}
           if exclude_prior_paths else {}),
        "dedup": {"combat_runs_distinct": int(data["_runs_distinct"][0]), "combat_runs_dropped": int(data["_runs_dropped"][0]),
                  "combat_samples_dropped": int(data["_samples_dedup_dropped"][0])},
        "samples_trained": int(data["lengths"].size), "target": "root visits (live search has no Gumbel root)" if policy_target == "visits" else policy_target,
        "validation": "by run (train_combat splits on the `fights` column, which holds the run index)",
        "build_s": round(time.perf_counter() - t0, 1)}
    doc["combat"] = train_combat_prior(data, cand, device=device, epochs=epochs, threads=threads, seed=gen,
                                       policy_target=policy_target)
    write_prior_model_json(cand, target)
    base_sims = int(mans[0][1]["base_sims"])
    if vrun:
        paths = [d / n for d, m in mans for n in m.get("run_shards", [])]
        doc["run_value"] = train_run_value(paths, cand, generation=gen, base_sims=base_sims,
                                           ascension=int(mans[0][1]["ascension"]), threads=threads, device=device)
        doc["dedup"]["run_records_dropped"] = int(doc["run_value"]["value"].get("duplicates_dropped", 0))
        doc["dedup"]["run_records_distinct"] = int(doc["run_value"]["value"].get("runs", 0))
    del rs
    if export:
        doc["onnx"] = export_all(cand)
    doc["cpu_min_total"] = round(doc["combat"]["cpu_min"] + doc.get("run_value", {}).get("cpu_min", 0.0), 2)
    doc["wall_min_total"] = round(doc["combat"]["wall_min"] + doc.get("run_value", {}).get("wall_min", 0.0), 2)
    # V_run drives macro choices once its keep conditions are shown (run_train.keep) and the switch is not opted out (D33); the
    # rules play outside fights otherwise. The record sits in train.json under vrun_keep; the agent.json macro follows its verdict.
    keep_doc = (doc.get("run_value", {}).get("value") or {}).pop("keep", None)
    doc["vrun_keep"] = keep_doc if keep_doc is not None else {"status": "not_run", "all_computable_pass": False}
    passed, enabled = vrun_keep_passed(doc["vrun_keep"], cand), macro_vrun_on()
    fit = doc["vrun_fit"] = carry_vrun_fit(cand, doc["vrun_keep"], vrun_seed) if vrun else {"source": "none", "kept": False, "sha256": None}
    macro = macro_for({"all_computable_pass": fit["kept"]}, cand, enabled)  # a kept fit (new or seed) and the export decide
    doc["vrun_keep"]["macro_switch"] = {"passed": passed, "enabled": enabled, "carried": fit["source"], "note": MACRO_SWITCH_NOTE}
    doc["vrun_keep"]["note"] = vrun_keep_note(macro)
    lineage = lineage_note(gen, target, macro, fit["source"])
    if lineage:
        doc["lineage"] = lineage
    write_json(cand / "agent.json", {"generation": target, "macro": macro, "trained_from": gen, "written": utc_now(),
                                     **({"lineage": lineage} if lineage else {}),
                                     "mask_version": CANDIDATE_MASK_VERSION, "vrun_fit": {"source": fit["source"], "sha256": fit["sha256"]},
                                     "note": ("V_run passed the computable keep conditions (Tier-B regret and the battery are not "
                                              "computed) and the macro switch is on (D33): it picks macro options" if macro == "vrun" and fit["source"] == "new" else
                                              "the seed V_run fit (the A/B's) is carried: the new fit did not pass the keep test (D33)" if macro == "vrun" else
                                              f"V_run passed the computable keep conditions; {MACRO_VRUN_FLAG}=rules opts "
                                              "out, so macro stays rules" if passed else
                                              "macro stays rules: V_run did not pass the keep conditions")})
    write_json(cand / "train.json", doc)
    shutil_rmtree(cand / "work")
    return cand


def vrun_keep_note(macro: str) -> str:
    """train.json's vrun_keep note: what the macro verdict means for the candidate's E-block."""
    if macro == "vrun":
        return ("V_run passed the computable keep conditions only (Tier-B regret and the battery are not computed); the candidate "
                "plays a new combat prior and V_run's macro together, so its E-block judges them together and a gain or a loss "
                "cannot be attributed to one of them")
    return "macro stays rules: the E-block judges the combat prior alone"


def shutil_rmtree(p: Path) -> None:
    import shutil  # noqa: PLC0415

    shutil.rmtree(p, ignore_errors=True)


# ------------------------------------------------------------------ lab markers


_MARKER = re.compile(r"(ready|done)-(\d+)(?:-r(\d+))?")
_MARKER_TARGET = re.compile(r"(ready|done)-(\d+)-g(\d+)")  # names the candidate generation; not counted as a budget by mark_ready


def marker_parts(path: Path | str) -> tuple[int, int] | None:
    """(generation, retrain index) of `ready-<gen>` (index 0), `ready-<gen>-r<n>` or `ready-<gen>-g<target>` (index 0: it is
    written as gen<target>/candidate); None for anything else. `marker_target` gives the `-g` form's target."""
    name = Path(path).name
    m = _MARKER.fullmatch(name)
    if m and m[1] == "ready":
        return int(m[2]), int(m[3] or 0)
    t = _MARKER_TARGET.fullmatch(name)
    return (int(t[2]), 0) if t and t[1] == "ready" else None


def marker_target(path: Path | str) -> int | None:
    """The candidate generation a `ready-<gen>-g<target>` marker names; None for the other forms."""
    t = _MARKER_TARGET.fullmatch(Path(path).name)
    return int(t[3]) if t and t[1] == "ready" else None


def marker_key(path: Path | str) -> tuple[int, int, int] | None:
    """Marker order: generation, then the plain and `-r<n>` forms by retrain index, then the `-g<target>` forms by target."""
    parts = marker_parts(path)
    if parts is None:
        return None
    target = marker_target(path)
    return (parts[0], 0, parts[1]) if target is None else (parts[0], 1, target)


def ready_markers(train_dir: Path) -> list[tuple[int, Path]]:
    """`LAB/train/ready-<gen>[-r<n>|-g<target>]` markers, lowest generation first, within one: the plain, the `-r<n>`, the `-g<target>`."""
    d = Path(train_dir)
    found = [(marker_key(p), p) for p in (d.glob("ready-*") if d.is_dir() else [])]
    return [(k[0], p) for k, p in sorted(((k, p) for k, p in found if k), key=lambda x: x[0])]


def rejections(tsv: Path | None, cand_gen: int) -> int:
    """How many candidates of generation `cand_gen` were rejected so far."""
    return sum(1 for r in read_promotions(tsv) if r.get("cand_gen") == str(cand_gen) and r.get("decision") == "rejected")


def mark_ready(lab: Path, gen: int, shards_root: Path, budget: int = BUDGET_RUNS, tsv: Path | None = None,
               review_guard: bool | None = None) -> Path | None:
    """Write `LAB/train/ready-<gen>` once the generation's self-play budget is spent, never twice (ready- or done-).

    After a rejection the generation gets another budget: once its candidate was rejected (a rejected row for generation
    gen + 1 in `promotions.tsv`) and the generation's runs reach (markers so far + 1) x budget, `ready-<gen>-r<n>` follows, and
    `train` retrains on the larger window. A promotion moves self-play to the next generation, so no further marker appears.
    With the review guard on (`review_guard`, default the SPIRELINE_REVIEW_GUARD variable) a generation gets at most
    `promotion_guard.ATTEMPT_CAP` budgets: no marker after its second rejection.
    """
    d = Path(lab) / "train"
    issued = sum(1 for p in (d.glob("*") if d.is_dir() else []) if (m := _MARKER.fullmatch(p.name)) and int(m[2]) == gen)
    if issued and rejections(tsv, gen + 1) < issued:
        return None
    if issued and review_guard_on(review_guard) and issued >= review_guard_module().ATTEMPT_CAP:
        return None  # the review guard's cap: no third promotion attempt (budget) for this generation
    if runs_in_generation(shards_root, gen) < budget * (issued + 1):
        return None
    d.mkdir(parents=True, exist_ok=True)
    p = d / (f"ready-{gen}" if issued == 0 else f"ready-{gen}-r{issued}")
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


def decide(floor: dict, boss: dict, wins: dict, cand_win_rate: float, base_win_rate: float, alpha: float = ALPHA,
           cand_boss_rate: float | None = None, base_boss_rate: float | None = None) -> dict[str, Any]:
    """The promotion rule (DECISIONS D24): the paired floor's lower bound above 0 (primary). The Act-1 boss kill is a
    not-worse guard while either arm kills the boss in under 10% of runs (it fails only when the exact McNemar says the
    candidate is worse at alpha) and a co-primary once both reach 10% (McNemar below alpha with the candidate ahead).
    Wins join as a co-primary once both arms win at least 10% of runs. Boss rates default to the counts in `boss`."""
    pairs = boss.get("pairs") or 0
    cbr = cand_boss_rate if cand_boss_rate is not None else (boss["cand"] / pairs if pairs and "cand" in boss else 0.0)
    bbr = base_boss_rate if base_boss_rate is not None else (boss["base"] / pairs if pairs and "base" in boss else 0.0)
    boss_co_primary = cbr >= WIN_GATE and bbr >= WIN_GATE
    p_worse = mcnemar_exact_one_sided(boss["c"], boss["b"])
    if boss_co_primary:
        boss_ok = boss["p"] < alpha and boss["b"] > boss["c"]
    else:
        boss_ok = not (p_worse < alpha and boss["c"] > boss["b"])
    checks = {"floor": bool(floor["lower"] > 0), "boss_kill": bool(boss_ok)}
    wins_active = cand_win_rate >= WIN_GATE and base_win_rate >= WIN_GATE
    if wins_active:
        checks["wins"] = bool(wins["p"] < alpha and wins["b"] > wins["c"])
    return {"checks": checks, "wins_co_primary": wins_active, "boss_kill_mode": "co-primary" if boss_co_primary else "guard",
            "boss_kill_p_worse": p_worse, "promote": all(checks.values()), "alpha": alpha, "rule": "D24"}


# ------------------------------------------------------------------ the review's guard (behind the flag)


def review_guard_module():
    """tools/lab/promotion_guard, imported on demand: the pure guard module (docs/PROMOTION_GUARD.md)."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools.lab import promotion_guard  # noqa: PLC0415

    return promotion_guard


def review_guard_on(flag: bool | None = None) -> bool:
    """Whether the review's guard decides: `flag` when given, else the SPIRELINE_REVIEW_GUARD variable (off by default).
    The variable is read here first and the guard module is imported only when it is on, so the flag off never imports
    it or puts the repo root on sys.path (the training venv's); the module's `enabled` still has the last word."""
    if flag is not None:
        return bool(flag)
    value = os.environ.get(REVIEW_GUARD_FLAG, "")
    return value.strip().lower() in REVIEW_GUARD_ON_VALUES and review_guard_module().enabled(value)


def lineage_blocks(rows: list[dict[str, str]], blocks: int = 3) -> list[tuple[int, int]]:
    """The promoted lineage's (kills, runs) on the last `blocks` judged blocks, oldest first: the candidate's kills on a
    block that promoted it, the incumbent's where it stayed. A block judged again (`rejudge` adds a second row for it) counts
    once, by its last row, in the place of its first judgement. Rows without the counts are skipped."""
    by_block: dict[Any, tuple[int, int]] = {}
    for i, r in enumerate(rows):
        try:
            n = int(r.get("seeds") or 0)
            k = int(r.get("boss_cand") if r.get("decision") == "promoted" else r.get("boss_base"))  # type: ignore[arg-type]
        except (TypeError, ValueError):
            continue
        if n > 0:
            by_block[r.get("seed_block") or ("row", i)] = (k, n)  # a row naming no block stands alone
    return list(by_block.values())[-blocks:]


def attempts_before(rows: list[dict[str, str]], cand_gen: int, cand_sha: str) -> int:
    """Other candidates of generation `cand_gen` judged so far (distinct shas; a row without a sha counts as one)."""
    return len({r.get("cand_sha", "") for r in rows if r.get("cand_gen") == str(cand_gen) and r.get("cand_sha", "") != cand_sha})


def previous_drift_flagged(rows: list[dict[str, str]]) -> bool:
    """Whether the last judged candidate's drift check flagged (the drift column; empty while the guard was off)."""
    return bool(rows) and rows[-1].get("drift", "") in ("flagged", "blocked")


def review_guard_verdict(cand_rows: list[dict], base_rows: list[dict], rows: list[dict[str, str]], *, cand_gen: int, cand_sha: str,
                         block: str, boot: int = BOOT_B, apply_cap: bool = True) -> dict[str, Any]:
    """The review's guard (promotion_guard.judge) on two arms, with the lineage, the drift history and the attempt count
    from the promotions.tsv rows written before this judgement (`rows`; this block's own rows do not count as lineage).
    With `apply_cap` false (`rejudge`: a re-read is not a new attempt) the attempt is reported against the cap but allowed."""
    pg = review_guard_module()
    kills_c, kills_b, dmg_c, dmg_b = pg.from_rows(cand_rows, base_rows, kill=boss_kill)
    history = [r for r in rows if r.get("seed_block") != block]
    prior = attempts_before(rows, cand_gen, cand_sha)
    cap = pg.ATTEMPT_CAP if apply_cap else max(pg.ATTEMPT_CAP, prior + 1)  # a cap the attempt cannot exceed blocks nothing
    verdict = pg.judge(kills_c, kills_b, dmg_c, dmg_b, lineage=lineage_blocks(history), previous_flagged=previous_drift_flagged(history),
                       prior_attempts=prior, cap=cap, boot=boot)
    verdict["attempt"]["cap"] = pg.ATTEMPT_CAP  # report the real cap ("attempt 3/2"), allowed or not
    return verdict


def decide_review(floor: dict, boss: dict, wins: dict, cand_win_rate: float, base_win_rate: float, guard: dict[str, Any],
                  alpha: float = ALPHA) -> dict[str, Any]:
    """D24 with the review's guard (docs/PROMOTION_GUARD.md): the floor primary and the 10% co-primary clauses as `decide`;
    in guard mode the boss kill is the review's McNemar guard at its own alpha (0.05); the boss-damage guard, the drift
    check's block and the attempt cap are further not-worse checks. A flagged (not blocked) drift sets `review`."""
    d = decide(floor, boss, wins, cand_win_rate, base_win_rate, alpha)
    checks = dict(d["checks"])
    if d["boss_kill_mode"] == "guard":
        checks["boss_kill"] = not guard["kill"]["fires"]
    checks["boss_damage"] = not guard["damage"]["fires"]
    checks["drift"] = not guard["drift"]["blocks"]
    checks["attempt"] = bool(guard["attempt"]["allowed"])
    return {**d, "checks": checks, "promote": all(checks.values()), "rule": "D24+" + guard["rule"], "guard_alpha": guard["alpha"],
            "review": bool(guard["review"])}


def arm_rows(arm_dir: Path) -> list[dict]:
    return [json.loads(x) for x in (Path(arm_dir) / "runs.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def arm_summary(rows: list[dict]) -> dict[str, Any]:
    import gen0  # noqa: PLC0415

    t = gen0.engine_block(rows)
    k = sum(boss_kill(r) for r in rows)
    lo, hi = gen0.wilson(k, len(rows))
    return {**t, "boss_kills": k, "boss_kill_rate": k / len(rows) if rows else None, "boss_kill_lo": lo, "boss_kill_hi": hi}


def compare_arms(cand_rows: list[dict], base_rows: list[dict], alpha: float = ALPHA, boot: int = BOOT_B,
                 guard: dict[str, Any] | None = None) -> dict[str, Any]:
    """The promotion table's numbers from two arms' rows, paired on seed. With `guard` (the review's verdict,
    `review_guard_verdict`) the decision is `decide_review`'s and the verdict is kept under `review_guard`; without, D24's."""
    c = {r["seed"]: r for r in cand_rows}
    b = {r["seed"]: r for r in base_rows}
    seeds = sorted(set(c) & set(b))
    diffs = [float(c[s]["floor"]) - float(b[s]["floor"]) for s in seeds]
    floor = cluster_bootstrap_lower(diffs, alpha, boot)
    boss = paired_binary({s: boss_kill(c[s]) for s in seeds}, {s: boss_kill(b[s]) for s in seeds})
    wins = paired_binary({s: bool(c[s]["win"]) for s in seeds}, {s: bool(b[s]["win"]) for s in seeds})
    cw = sum(bool(c[s]["win"]) for s in seeds) / max(1, len(seeds))
    bw = sum(bool(b[s]["win"]) for s in seeds) / max(1, len(seeds))
    out: dict[str, Any] = {"paired_seeds": len(seeds), "floor": floor, "boss_kill": boss, "wins": wins}
    if guard is None:
        return {**out, "decision": decide(floor, boss, wins, cw, bw, alpha)}
    return {**out, "review_guard": guard, "decision": decide_review(floor, boss, wins, cw, bw, guard, alpha)}


# ------------------------------------------------------------------ the E-block and the promotion run


TSV_COLUMNS = ("written", "cand_gen", "base_gen", "seed_block", "seeds", "d_floor", "floor_lower", "p_floor", "boss_cand",
               "boss_base", "p_boss", "wins_cand", "wins_base", "decision", "cand_sha", "cand_dir",
               # the review's guard: the attempt of the generation (always), the drift verdict and the guard line (empty / "off" when off)
               "attempt", "drift", "guard")


def guard_cells(guard: dict[str, Any] | None, attempt: int) -> dict[str, Any]:
    """The three guard columns of a promotions.tsv row."""
    if guard is None:
        return {"attempt": attempt, "drift": "", "guard": "off"}
    return {"attempt": attempt, "drift": guard["drift"]["verdict"], "guard": review_guard_module().summary(guard)}


def read_promotions(tsv: Path | None) -> list[dict[str, str]]:
    """The rows of promotions.tsv by column name (rows written before the sha column have none: ``""``)."""
    if tsv is None or not Path(tsv).is_file():
        return []
    lines = [x for x in Path(tsv).read_text(encoding="utf-8").splitlines() if x.strip()]
    head = lines[0].split("\t") if lines else []
    return [dict(zip(head, x.split("\t"), strict=False)) for x in lines[1:]]


def append_promotion(tsv: Path, values: dict[str, Any]) -> None:
    """Add a row; a file with an older header is rewritten to the current columns first (old rows keep their values)."""
    tsv = Path(tsv)
    rows = read_promotions(tsv)
    if not tsv.is_file() or tsv.read_text(encoding="utf-8").splitlines()[0].split("\t") != list(TSV_COLUMNS):
        tsv.write_text("\t".join(TSV_COLUMNS) + "\n" + "".join("\t".join(r.get(c, "") for c in TSV_COLUMNS) + "\n" for r in rows),
                       encoding="utf-8")
    with tsv.open("a", encoding="utf-8") as fh:
        fh.write("\t".join(str(values.get(c, "")) for c in TSV_COLUMNS) + "\n")


def block_used_by_another(tsv: Path | None, block: str, cand_sha: str) -> bool:
    """Whether `block` already judged a candidate other than `cand_sha` (a row without a sha counts as another one)."""
    return any(r.get("seed_block") == block and r.get("cand_sha", "") != cand_sha for r in read_promotions(tsv))


def check_disjoint(seeds: list[str], others: dict[str, list[str]]) -> None:
    """Raise if the block shares a seed with any of the named seed collections."""
    mine = set(seeds)
    for name, other in others.items():
        both = mine & set(other)
        if both:
            raise ValueError(f"{len(both)} seeds also in {name}")


COMMITTED_SEEDS = ROOT / "referee" / "seeds"  # the benchmark blocks the checkout carries; nothing generated is written here


def lab_seed_dir(lab: Path | str) -> Path:
    """Where the loop's generated E-blocks live: `<lab>/seeds`, outside the checkout, so the tree stays clean (the presort's
    refset score refuses a tree with uncommitted or untracked files)."""
    return Path(lab) / "seeds"


def known_seed_sets(seed_dir: Path, lab_runs: Path | list[Path] | None, e_name: str,
                    extra_dirs: Sequence[Path | str] = ()) -> dict[str, list[str]]:
    """Every seed block except this one, and the seeds the lab's self-play blocks actually played. The blocks are read from
    `seed_dir`, from the checkout's committed folder and from each of `extra_dirs` (the lab's generated blocks), so a block
    is checked against every other whichever folder holds it."""
    import gen0  # noqa: PLC0415

    out: dict[str, list[str]] = {}
    folders: dict[Path, Path] = {}
    for d in (Path(seed_dir), COMMITTED_SEEDS, *map(Path, extra_dirs)):
        folders.setdefault(d.resolve(), d)
    for d in folders.values():
        for f in sorted(d.glob("*.json")):
            if f.stem != e_name and f.stem not in out:
                out[f.stem] = gen0.load_seeds(f)
    for name in ("eval_seeds", "test_seeds"):  # run_train's held-out seeds (DESIGN 5.6)
        f = ROOT / "rl" / "run_train" / f"{name}.json"
        if f.is_file():
            out[f"run_train {name}"] = gen0.load_seeds(f)
    sp: list[str] = []
    for d in ([lab_runs] if isinstance(lab_runs, (str, Path)) else list(lab_runs or [])):
        for f in Path(d).glob("*/runs.jsonl"):
            with contextlib.suppress(OSError, ValueError):
                sp += [json.loads(x)["seed"] for x in f.read_text(encoding="utf-8").splitlines() if x.strip()]
    if lab_runs:
        out["self-play runs"] = sp
    return out


def run_arm(seeds: list[str], arm_dir: Path, agent: Agent, config: str, procs: int, ascension: int = 10,
            server_opts=None) -> list[dict]:
    """Play the seeds with `agent`, or reuse the arm if the same seeds and config were already played there. The arm uses a
    prior server only when `server_opts` (a `prior_launch.ServerOpts`) says so; the promotion arms default to the CPU path."""
    import gen0  # noqa: PLC0415

    arm_dir = Path(arm_dir)
    s = arm_dir / "summary.json"
    if s.is_file():
        old = read_json(s)
        rows = arm_rows(arm_dir)
        wanted = "server" if (server_opts is not None and server_opts.mode != "off") else "cpu"
        played = "server" if old.get("prior_server") else "cpu"  # both arms of one comparison play on one transport
        if (old.get("config") == config and old.get("agent") == agent.describe() and {r["seed"] for r in rows} == set(seeds)
                and played == wanted):
            return rows
    if server_opts is not None and server_opts.mode != "off":
        gen0.run_block_served(seeds, config, procs, arm_dir, opts=server_opts, ascension=ascension, agent=agent)
    else:
        gen0.run_block(seeds, config, procs, arm_dir, ascension=ascension, agent=agent)
    return arm_rows(arm_dir)


def promote(cand: Path, base: Path | None, seeds_file: Path, *, agent_root: Path, procs: int = 16, config: str = "a3-v1",
            ascension: int = 10, lab: Path | None = None, lab_runs: list[Path] | None = None, boot: int = BOOT_B,
            review_guard: bool | None = None, server_opts=None) -> dict[str, Any]:
    """Play candidate and base on the same E-block seeds, test, and record the decision. With the review guard on
    (`review_guard`, default the SPIRELINE_REVIEW_GUARD variable; off) the decision is `decide_review`'s, a third attempt
    of a generation is refused before any run is played, and the tsv row carries the attempt and the guard's verdict.
    `server_opts` puts both arms on a prior server each (one policy per server); the default is the CPU path."""
    import gen0  # noqa: PLC0415

    seeds = gen0.load_seeds(seeds_file)
    name = Path(seeds_file).stem
    if name.startswith("SP-"):
        raise ValueError("an E-block is never named like a self-play stream")
    check_disjoint(seeds, known_seed_sets(Path(seeds_file).parent, lab_runs, name, [lab_seed_dir(lab)] if lab else ()))
    cand_agent = load_agent(cand)
    base_agent = load_agent(base, agent_root=None if base else agent_root)
    cand_sha = cand_agent.describe()["prior_sha256"] or "none"
    tsv = Path(agent_root) / "promotions.tsv"
    if block_used_by_another(tsv, name, cand_sha):  # a second look at one block is a look the alpha does not pay for
        raise ValueError(f"E-block {name} already judged another candidate; make a fresh one (resolve_e_block)")
    rows = read_promotions(tsv)
    guard_on = review_guard_on(review_guard)
    attempt = attempts_before(rows, cand_agent.generation, cand_sha) + 1
    if guard_on and attempt > review_guard_module().ATTEMPT_CAP:
        raise ValueError(f"generation {cand_agent.generation} already had {attempt - 1} promotion attempts "
                         f"(the review guard caps them at {review_guard_module().ATTEMPT_CAP}); no block is played")
    arm = f"{name}-{config}"
    arm_kw = {"server_opts": server_opts} if server_opts is not None else {}  # the CPU path passes nothing new
    cand_rows = run_arm(seeds, Path(cand) / "arms" / arm, cand_agent, config, procs, ascension, **arm_kw)
    base_dir = Path(base_agent.path) if base_agent.path else Path(agent_root) / "gen0"
    base_rows = run_arm(seeds, base_dir / "arms" / arm, base_agent, config, procs, ascension, **arm_kw)
    guard = (review_guard_verdict(cand_rows, base_rows, rows, cand_gen=cand_agent.generation, cand_sha=cand_sha, block=name, boot=boot)
             if guard_on else None)
    res = compare_arms(cand_rows, base_rows, ALPHA, boot, guard)
    if res["paired_seeds"] != len(seeds):  # a partial arm must not promote on fewer pairs than the block
        raise ValueError(f"only {res['paired_seeds']} of {len(seeds)} seeds are paired in both arms")
    doc = {"written": utc_now(), "seed_block": name, "n_seeds": len(seeds), "config": config, "procs": procs,
           "alpha": ALPHA, "cand": {"agent": cand_agent.describe(), "summary": arm_summary(cand_rows)},
           "base": {"agent": base_agent.describe(), "summary": arm_summary(base_rows)}, **res,
           "decision_text": "promoted" if res["decision"]["promote"] else "rejected", "cand_sha": cand_sha,
           # DESIGN 5.3/5.4 ask for both; PLAN section 5 does not, so a promotion here is not the full DESIGN gate
           "f7": "not run", "matched_wall_clock": False}
    write_json(Path(cand) / "promotion.json", doc)
    f, bk = doc["floor"], doc["boss_kill"]
    append_promotion(tsv, {"written": doc["written"], "cand_gen": cand_agent.generation, "base_gen": base_agent.generation,
                           "seed_block": name, "seeds": doc["n_seeds"], "d_floor": f"{f['mean']:.4f}", "floor_lower": f"{f['lower']:.4f}",
                           "p_floor": f"{f['p']:.5f}", "boss_cand": bk["cand"], "boss_base": bk["base"], "p_boss": f"{bk['p']:.5f}",
                           "wins_cand": doc["cand"]["summary"]["wins"], "wins_base": doc["base"]["summary"]["wins"],
                           "decision": doc["decision_text"], "cand_sha": cand_sha, "cand_dir": Path(cand).name,
                           **guard_cells(guard, attempt)})
    if res["decision"]["promote"]:
        write_json(Path(agent_root) / "current.json", {"generation": cand_agent.generation, "dir": str(Path(cand).resolve()),
                                                        "promoted": doc["written"], "seed_block": name})
        if lab is not None:
            p = Path(lab) / "score" / "pending"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(utc_now() + "\n", encoding="utf-8")
    return doc


def rejudge(cand: Path, base: Path | None, seeds_file: Path, *, agent_root: Path, config: str = "a3-v1", note: str,
            lab: Path | None = None, boot: int = BOOT_B, review_guard: bool | None = None) -> dict[str, Any]:
    """Apply the current rule to a candidate's finished arms on its own block, without playing again: for an owner
    decision that changes the rule after a block was judged (D24). Both cached arms must cover the whole block. The
    earlier promotion.json is kept beside the new one; the tsv gets a second row for the same candidate and block.
    `review_guard` as in `promote` (a re-read is not a new attempt: the attempt number is reported against the cap,
    never applied here; the other checks decide)."""
    import gen0  # noqa: PLC0415

    seeds = gen0.load_seeds(seeds_file)
    name = Path(seeds_file).stem
    cand_agent = load_agent(cand)
    base_agent = load_agent(base, agent_root=None if base else agent_root)
    arm = f"{name}-{config}"
    base_dir = Path(base_agent.path) if base_agent.path else Path(agent_root) / "gen0"
    cand_rows, base_rows = arm_rows(Path(cand) / "arms" / arm), arm_rows(base_dir / "arms" / arm)
    cand_sha = cand_agent.describe()["prior_sha256"] or "none"
    rows = read_promotions(Path(agent_root) / "promotions.tsv")
    attempt = attempts_before(rows, cand_agent.generation, cand_sha) + 1
    guard = (review_guard_verdict(cand_rows, base_rows, rows, cand_gen=cand_agent.generation, cand_sha=cand_sha, block=name, boot=boot,
                                  apply_cap=False) if review_guard_on(review_guard) else None)
    res = compare_arms(cand_rows, base_rows, ALPHA, boot, guard)
    if res["paired_seeds"] != len(seeds):
        raise ValueError(f"only {res['paired_seeds']} of {len(seeds)} seeds are paired in both cached arms")
    old = Path(cand) / "promotion.json"
    before = json.loads(old.read_text(encoding="utf-8")) if old.is_file() else None
    if before is not None:
        old.rename(Path(cand) / f"promotion-{before.get('decision_text', 'earlier')}-{str(before.get('written', 'x'))[:10]}.json")
    doc = {"written": utc_now(), "seed_block": name, "n_seeds": len(seeds), "config": config, "alpha": ALPHA,
           "cand": {"agent": cand_agent.describe(), "summary": arm_summary(cand_rows)},
           "base": {"agent": base_agent.describe(), "summary": arm_summary(base_rows)}, **res,
           "decision_text": "promoted" if res["decision"]["promote"] else "rejected", "cand_sha": cand_sha,
           "rejudged": {"note": note, "earlier_decision": (before or {}).get("decision_text")},
           "f7": "not run", "matched_wall_clock": False}
    write_json(Path(cand) / "promotion.json", doc)
    f, bk = doc["floor"], doc["boss_kill"]
    append_promotion(Path(agent_root) / "promotions.tsv",
                     {"written": doc["written"], "cand_gen": cand_agent.generation, "base_gen": base_agent.generation,
                      "seed_block": name, "seeds": doc["n_seeds"], "d_floor": f"{f['mean']:.4f}", "floor_lower": f"{f['lower']:.4f}",
                      "p_floor": f"{f['p']:.5f}", "boss_cand": bk["cand"], "boss_base": bk["base"], "p_boss": f"{bk['p']:.5f}",
                      "wins_cand": doc["cand"]["summary"]["wins"], "wins_base": doc["base"]["summary"]["wins"],
                      "decision": doc["decision_text"], "cand_sha": cand_sha, "cand_dir": Path(cand).name, **guard_cells(guard, attempt)})
    if res["decision"]["promote"]:
        write_json(Path(agent_root) / "current.json", {"generation": cand_agent.generation, "dir": str(Path(cand).resolve()),
                                                        "promoted": doc["written"], "seed_block": name, "note": note})
        if lab is not None:
            p = Path(lab) / "score" / "pending"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(utc_now() + "\n", encoding="utf-8")
    return doc


def resolve_e_block(template: str, k: int, n: int = 2000, *, tsv: Path | None = None, cand_sha: str | None = None,
                    also_dirs: Sequence[Path | str] | None = None) -> Path:
    """The E-block for candidate generation `k`: `template` with {k} for the generation and {date} for today (UTC).
    An existing `E<k>-*.json` that has not judged another candidate (per `promotions.tsv`) is used as it is
    (E1-2026-10-23 is committed). If every such block did, a new `E<k>-<date>-r<n>` is created by the seed rule, so a
    candidate is never judged on a block that already judged a different one. A new block is written to the template's
    folder (the lab's `seeds` folder, outside the checkout); existing blocks are looked for there and in `also_dirs`
    (default: the checkout's committed folder), and a new name is free in all of them."""
    import gen0  # noqa: PLC0415

    path = Path(template.format(k=k, date=datetime.now(UTC).strftime("%Y-%m-%d")))
    found: dict[str, Path] = {}
    for d in (path.parent, *map(Path, [COMMITTED_SEEDS] if also_dirs is None else also_dirs)):
        for p in sorted(d.glob(f"E{k}-*.json")):
            found.setdefault(p.stem, p)
    existing = [found[name] for name in sorted(found)]
    for p in existing:
        if not block_used_by_another(tsv, p.stem, cand_sha or "none"):
            return p
    if existing:
        j = 1
        while f"{path.stem}-r{j}" in found or (path.parent / f"{path.stem}-r{j}.json").exists():
            j += 1
        path = path.parent / f"{path.stem}-r{j}.json"
    gen0.write_seed_file(path.stem, n, path)
    return path


def run_promote_subprocess(emu_python: str, cand: Path, base: Path | None, seeds_file: Path, agent_root: Path, procs: int,
                           lib: str | None, lab: Path | None, lab_runs: Path | None = None, review_guard: bool = False,
                           prior_server_flags: tuple[str, ...] = ()) -> int:
    """From the train job (GPU venv): promotion plays emulators, which run in the emulator venv. The child inherits the
    environment, so the SPIRELINE_REVIEW_GUARD variable reaches it; `review_guard` passes the explicit flag on, and
    `prior_server_flags` the server options (none by default: the arms play on the CPU path)."""
    cmd = [emu_python, str(ROOT / "rl" / "gen0.py"), "promote", "--cand", str(cand), "--seeds-file", str(seeds_file),
           "--agent-root", str(agent_root), "--procs", str(procs)]
    if base:
        cmd += ["--base", str(base)]
    if lib:
        cmd += ["--lib", lib]
    if lab:
        cmd += ["--lab", str(lab)]
    if lab_runs:
        cmd += ["--lab-runs", str(lab_runs)]  # the E-block must not hold a seed the self-play streams played
    if review_guard:
        cmd += ["--review-guard"]
    cmd += list(prior_server_flags)  # none by default: the promotion arms play on the CPU path unless a flag says otherwise
    return subprocess.run(cmd, check=False).returncode


__all__ = [n for n in dir() if not n.startswith("_")]
_ = dataclasses
