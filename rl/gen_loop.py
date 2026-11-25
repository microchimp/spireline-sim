"""The expert-iteration generation loop (docs/PLAN.md section 5 Lane A, docs/DESIGN.md 5.3-5.4).

    python rl/gen0.py emu --stream JOB --config a3-v1 --record --agent-root AGENT --out JOBDIR   # self-play with samples
    python rl/gen0.py train --gen N --shards-root RUNS --agent-root AGENT                         # -> AGENT/gen<N+1>/candidate
    python rl/gen0.py promote --cand AGENT/gen<N+1>/candidate --base AGENT/gen<N>/...             # E1 block, both arms
    python rl/gen0.py build --prior DIR --agent-root AGENT --candidate-generation K --why TEXT --marker-dir LAB/train
                                      # -> AGENT/gen<K>/candidate built, not trained; the lab's train job then judges it as it is

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
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field
from datetime import UTC, datetime
from functools import lru_cache
from pathlib import Path
from statistics import NormalDist
from typing import Any

import numpy as np

import fight_rows
import macro_recipe

ROOT = Path(__file__).resolve().parents[1]
DEFAULT_AGENT_ROOT = ROOT.parents[1] / "referee" / "agent"
GEN_WINDOW = 3  # DESIGN 5.4: train on the last 3 generations
BUDGET_RUNS = 2000  # self-play runs per generation before its train marker
# DESIGN 5.4: fights sampled for the combat net at a candidate's first try (x2, x3 at its retrains). 6,400 until 2026-11-02, when it
# drew 5 to 13% of the 75,000 to 120,000 fights a candidate has (D44 finding 4). The train job's slot (LAB/jobs.json train
# timeout_min 330) bounds it: about 32 ms of GPU a fight over 16 epochs (train.json combat.wall_min, gen 9 and 10), so the third
# try's 96,000 fights take about 51 min, and the slowest train job seen (11,365 s, plus 13 min more V_run fit at the third try)
# ends near 248 min, 82 min inside the slot.
MAX_FIGHTS = 32000
PART_RUNS = 50
# A run that finished waits in the sink for the rest of its part. A block the lab stops writes nothing after its last flush, so a part
# that fills slowly (the filler plays a few runs a minute beside the jobs) would lose up to PART_RUNS - 1 finished runs at every stop:
# 1,828 of the filler's 4,496 runs by 2026-11-04. A part is also written when its oldest run has waited this long.
FLUSH_AGE_S = 120.0
ACT1_LAST_FLOOR = 17  # the Act 1 boss is floor 17; a run past it killed the boss
ALPHA = 0.01
WIN_GATE = 0.10  # wins join the co-primaries once both arms reach this rate
BOOT_B = 20000
ENV_GEN_DIR = "SPIRELINE_GEN_DIR"
REVIEW_GUARD_FLAG = "SPIRELINE_REVIEW_GUARD"  # tools/lab/promotion_guard.FLAG: the review's guard decides when set; off by default
REVIEW_GUARD_ON_VALUES = ("1", "on", "true", "yes")  # tools/lab/promotion_guard.ON_VALUES (a test keeps them equal)
KEEP_RULE_FLAG = "SPIRELINE_KEEP_RULE"  # the promotion rule: "v1" (the floor primary, the default), "v2" (tools/lab/keep_v2.py, docs/rl/KEEP_TEST_V2.md) or "v3" (D66, docs/rl/KEEP_TEST_V3.md)
KEEP_RULES = ("v1", "v2", "v3")
MIN_GAIN_DEFAULT = 1.0  # route A's smallest deep_f gain per 100 runs when nothing says otherwise: `keep_v2.MIN_GAIN` (a test holds them equal), kept here so rule v1 never imports the module
KEEP_TESTS = ("v2", "v3")  # the rules that run through tools/lab/keep_v2.py (frontier, deep_f, the funnel's looks): v3 is a mode of v2's module
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
ROW_EXTRA_KEYS = fight_rows.ROW_EXTRA_KEYS  # fight_kind, grade, grade_sims: in a part only when its rows were recorded with a grade (D65)


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
    # a blended macro (`imitation_chooser`, macro "vrun+pi_h:lambda=...:pi_h=<8 hex>"): the human prior's weights (human_prior.npz in the
    # candidate's folder) and its lambda; the other macros leave them unset
    human_prior: str | None = None
    lam: float = 0.0
    # How this agent searches its fights beyond the lab's `--config` (2026-11-06, docs/rl/STAKES_AND_HOLD.md): a stakes schedule (agent.json
    # `search_schedule`, as its tag `<base config>@h<hall>e<elite>b<boss>`: the base config's simulations times a multiplier per fight kind;
    # the schedule's base replaces the config the lab passes) and the hallway potion hold (agent.json `potion_hold`, per cent of max HP).
    # Both are part of the agent's identity (`describe`, `recipe`): None is every agent that existed before them.
    stakes: str | None = None
    hold: int | None = None
    # lambda by act (2026-11-08, D55 addendum 12): the recipe's `:act<N>=<lambda>` parts as ((act number from 1, lambda), ...), sorted; () is
    # every agent that existed before them (the base `lam` for every act)
    lam_acts: tuple[tuple[int, float], ...] = ()
    # the run student (2026-11-11, `rl/run_student.py`, macro "vrun+pi_s:..."): its weights file (run_student.npz in the candidate's folder) and
    # its lambda per kind, ((kind, (Act 1, Act 2, Act 3)), ...) in the recipe's order; unset for every agent that existed before it
    student: str | None = None
    student_lams: tuple[tuple[str, tuple[float, float, float]], ...] = ()
    # the Act 1 elite bonus (2026-11-14, D60 item 9; agent.json `route_elite`, the recipe's `+elite:<bonus>@<per cent>`): (floors per elite room on the best
    # path through an Act 1 route option, the HP per cent of max HP at or above which it applies). Part of the identity, like the schedule and the hold;
    # None is every agent that existed before it.
    route_elite: tuple[float, int] | None = None
    # the boss chain (2026-11-19, the Act 3 week, `rl/boss_chain.py`; agent.json `boss_chain`, the recipe's `+chain:<8 hex>@<k1>/<k2>/<k3>`): the chain file
    # (boss_chain.npz in the candidate's folder), its sha256 and its kappa per act (floors per nat of the log chance of getting past the Act 2 boss and the
    # first Act 3 boss added to each option's value in Acts 2 and 3). Part of the identity; None is every agent that existed before it.
    chain: str | None = None
    chain_sha256: str | None = None
    chain_kappa: tuple[float, float, float] | None = None

    def search_config(self, sims):
        """The `search_configs.SearchConfig` this agent plays when the lab asks for `sims` (a config name or a flat budget): `sims` itself
        for an agent without a stakes schedule or a hold, so every older agent searches as it always did."""
        import search_configs  # noqa: PLC0415 -- pure

        return search_configs.for_agent(sims, self.stakes, self.hold)

    def config_name(self, sims) -> str:
        """The name `search_config(sims)` carries (what an arm's summary names as its `config`): `sims` as given for a plain agent."""
        if self.stakes is None and self.hold is None:
            return sims if isinstance(sims, str) else f"flat-{int(sims)}"
        return self.search_config(sims).name

    def wall_factor(self) -> float:
        """How many times the plain per-run wall limits (`gen0.RUN_TIMEOUT_S`, `RUN_WALL_CAP_S`, the real arm's run deadline) this agent's
        search is given: the stakes schedule's largest multiplier over four, never under 1."""
        return 1.0 if self.stakes is None else self.search_config("a3-v1").wall_factor()

    def recipe(self) -> str:
        """The ledger's recipe (`macro_recipe.full_recipe`): the macro's, the schedule's, the hold's, the elite bonus's and the boss chain's."""
        chain = (self.chain_sha256[:8], self.chain_kappa) if self.chain is not None else None
        return macro_recipe.full_recipe(self.macro, self.stakes, self.hold, self.route_elite, chain)

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
        if macro_recipe.is_blended(self.macro):
            if not (self.macro_npz and self.human_prior):
                raise ValueError(f"macro {self.macro} needs run_value.npz and human_prior.npz")
            import imitation_chooser  # noqa: PLC0415

            pick, bias = macro_recipe.parse_pick(self.macro)
            if macro_recipe.is_student(self.macro) and not self.student:
                raise ValueError(f"macro {self.macro} needs run_student.npz")
            return imitation_chooser.BlendedPolicy(self.macro_npz, self.human_prior, self.lam, self.tau, self.eps, self.W,
                                                   pick=pick, skip_bias=bias, act_lam=dict(self.lam_acts) or None,
                                                   **({"student": self.student, "student_lams": dict(self.student_lams)} if self.student else {}),
                                                   **({"route_elite": self.route_elite} if self.route_elite is not None else {}),
                                                   **({"chain": self.chain, "chain_kappa": self.chain_kappa} if self.chain is not None else {}))
        return selfplay.RulesPlayer()

    def describe(self) -> dict[str, Any]:
        sha = None
        if self.prior_dir:
            with contextlib.suppress(OSError, ValueError):
                sha = read_json(Path(self.prior_dir) / "model.json").get("policy_sha256")
        tail = "/".join(Path(self.path).parts[-2:]) if self.path else None
        # an exploring self-play agent (`gen0.py emu --explore-tau/--explore-eps`) names its sampling; the argmax agent every
        # evaluation plays keeps the description it always had
        explore = {k: v for k, v in (("tau", self.tau), ("eps", self.eps)) if v}
        # the search's own identity, named only where the agent has it, so an older agent's description is what it was
        search = {k: v for k, v in (("stakes", self.stakes), ("hold", self.hold)) if v is not None}
        if self.route_elite is not None:  # what the run chooser picks on the map, named only where the agent has it
            search["route_elite"] = [self.route_elite[0], self.route_elite[1]]
        if self.chain is not None:  # what the run chooser adds to its option values in Acts 2 and 3, named only where the agent has it
            search["boss_chain"] = macro_recipe.chain_tag(self.chain_sha256, self.chain_kappa)
        return {"generation": self.generation, "dir": tail, "prior": bool(self.prior_dir), "prior_sha256": sha,
                "macro": self.macro, **explore, **search}


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


def exploring_agent(agent: Agent | None, tau: float, eps: float) -> Agent:
    """The agent self-play records with: the same agent, its run-level choices sampled (DESIGN.md 5.5,
    `run_train.policy.LookaheadPolicy`: softmax over V_W at temperature `tau` floors, plus `eps` uniform) from each run's own
    exploration stream, so a run replays exactly. Only a vrun macro samples; evaluation (E-blocks, nights, scores) plays the
    argmax agent `load_agent` returns, never this one."""
    if agent is None or not (agent.macro == "vrun" or macro_recipe.is_blended(agent.macro)):
        raise ValueError("only a vrun macro explores: give the agent of a generation that plays V_run")
    if tau < 0 or not 0.0 <= eps <= 1.0:
        raise ValueError(f"tau must be >= 0 and eps in [0, 1], not {tau}, {eps}")
    return dataclasses.replace(agent, tau=float(tau), eps=float(eps))


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
    blended = macro_recipe.is_blended(mode)
    if (mode == "vrun" or blended) and not npz.is_file():
        raise FileNotFoundError(f"macro {mode} needs run_value.npz in {p.name}")
    human, lam, lam_acts = None, 0.0, {}
    if blended:  # the recipe names lambda and the prior's weights (8 hex of their sha256); the folder must hold exactly those
        lam, sha8 = macro_recipe.parse_macro(mode)
        lam_acts = macro_recipe.parse_act_lambdas(mode)
        named = meta.get("human_prior") or {}  # agent.json's own record of the prior: file, full sha256, lambda
        human = p / str(named.get("file") or "human_prior.npz")
        if not human.is_file():
            raise FileNotFoundError(f"macro {mode} needs {human.name} in {p.name}")
        sha = file_sha256(human)
        if sha[:8] != sha8 or (named.get("sha256") and named["sha256"] != sha):
            raise ValueError(f"{p.name}/{human.name} is not the prior macro {mode} names")
        if named.get("lambda") is not None and macro_recipe.format_lambda(lam) != str(named["lambda"]):
            raise ValueError(f"{p.name}/agent.json: the macro {mode} and the human_prior lambda {named['lambda']} disagree")
        if named.get("pick") is not None and named["pick"] != macro_recipe.parse_pick(mode)[0]:
            raise ValueError(f"{p.name}/agent.json: the macro {mode} and the human_prior pick {named['pick']} disagree")
        if named and named.get("lambda_acts", {}) != {str(a): macro_recipe.format_lambda(v) for a, v in sorted(lam_acts.items())}:
            raise ValueError(f"{p.name}/agent.json: the macro {mode} and the human_prior lambda_acts {named.get('lambda_acts')} disagree")
    student_file, student_lams = None, {}
    if macro_recipe.is_student(mode):  # the recipe names the student (8 hex of its sha256) and its lambda per kind and act
        sha8s, student_lams = macro_recipe.parse_student(mode)
        named_s = meta.get("student") or {}
        student_file = p / str(named_s.get("file") or STUDENT_FILE)
        if not student_file.is_file():
            raise FileNotFoundError(f"macro {mode} needs {student_file.name} in {p.name}")
        sha_s = file_sha256(student_file)
        if sha_s[:8] != sha8s or (named_s.get("sha256") and named_s["sha256"] != sha_s):
            raise ValueError(f"{p.name}/{student_file.name} is not the student macro {mode} names")
        want = {k: [macro_recipe.format_lambda(x) for x in v] for k, v in student_lams.items()}
        if named_s and named_s.get("kinds") != want:
            raise ValueError(f"{p.name}/agent.json: the macro {mode} and the student's kinds {named_s.get('kinds')} disagree")
        import human_prior_infer  # noqa: PLC0415 -- numpy only

        fitted = human_prior_infer.student_kinds_of(student_file)
        if any(k not in fitted for k in student_lams):
            raise ValueError(f"{p.name}/{student_file.name} was fitted for {sorted(fitted)}, the macro {mode} names {sorted(student_lams)}")
    mask_version = meta.get("mask_version", 1)
    if isinstance(mask_version, bool) or mask_version not in AGENT_MASK_VERSIONS:
        raise ValueError(f"{p.name}/agent.json: mask_version {mask_version!r} is not one of {list(AGENT_MASK_VERSIONS)}")
    stakes, hold = search_identity(meta, p.name)
    route_elite = route_elite_identity(meta, p.name)
    if route_elite is not None and not blended:
        raise ValueError(f"{p.name}/agent.json: route_elite needs a blended macro (the elite bonus rides on the run chooser), not {mode!r}")
    chain = boss_chain_identity(meta, p)
    if chain is not None and not blended:
        raise ValueError(f"{p.name}/agent.json: boss_chain needs a blended macro (the chain rides on the run chooser), not {mode!r}")
    agent = Agent(str(p), generation, prior, mode, str(npz) if mode == "vrun" or blended else None,
                  float(meta.get("tau", 0.0)), float(meta.get("eps", 0.0)), mask_version=int(mask_version),
                  human_prior=str(human) if human else None, lam=lam, stakes=stakes, hold=hold, lam_acts=tuple(sorted(lam_acts.items())),
                  student=str(student_file) if student_file else None, student_lams=tuple(student_lams.items()), route_elite=route_elite,
                  **({"chain": str(chain[0]), "chain_sha256": file_sha256(chain[0]), "chain_kappa": chain[1]} if chain is not None else {}))
    if meta.get("recipe") is not None and meta["recipe"] != agent.recipe():  # the recipe agent.json records is the one the fields give
        raise ValueError(f"{p.name}/agent.json: recipe {meta['recipe']!r} is not the agent's ({agent.recipe()!r})")
    return agent


def search_identity(meta: dict[str, Any], name: str = "agent.json") -> tuple[str | None, int | None]:
    """(stakes tag, hold per cent) of an agent.json: its `search_schedule` ({"base": a search config name, "mult": {"hall", "elite",
    "boss"}}) as the tag `<base>@h<hall>e<elite>b<boss>`, and its `potion_hold` ({"hp_frac": 0.30, "fights": "hall"}) as per cent of
    max HP; (None, None) for an agent with neither. A schedule whose base is not a search config, a multiplier that is not a whole
    number from 1 to 64, and a hold that is not a whole per cent are refused (the build and the load check alike)."""
    stakes = hold = None
    sched = meta.get("search_schedule")
    if sched is not None:
        import search_configs  # noqa: PLC0415 -- pure

        try:
            stakes = macro_recipe.stakes_tag(str(sched["base"]), sched["mult"])
            base = search_configs.get(str(sched["base"]))
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{name}: search_schedule {sched!r} is not a base config and a hall/elite/boss multiplier: {e}") from e
        if base.stakes is not None or base.potion_hold is not None:
            raise ValueError(f"{name}: search_schedule's base {sched['base']!r} already has a stakes schedule or a hold")
        if sched.get("tag") is not None and sched["tag"] != stakes:
            raise ValueError(f"{name}: search_schedule's tag {sched['tag']!r} is not {stakes!r}")
    ph = meta.get("potion_hold")
    if ph is not None:
        try:
            pct = round(float(ph["hp_frac"]) * 100)
            if abs(float(ph["hp_frac"]) * 100 - pct) > 1e-6:
                raise ValueError(f"hp_frac {ph['hp_frac']!r} is not a whole per cent")
            hold = macro_recipe.hold_pct(int(pct))
            if ph.get("fights", "hall") != "hall":
                raise ValueError(f"fights {ph['fights']!r}: the hold is for hallway fights")
        except (KeyError, TypeError, ValueError) as e:
            raise ValueError(f"{name}: potion_hold {ph!r} is not {{'hp_frac': <whole per cent as a fraction>, 'fights': 'hall'}}: {e}") from e
    return stakes, hold


def route_elite_identity(meta: dict[str, Any], name: str = "agent.json") -> tuple[float, int] | None:
    """(bonus, HP per cent) of an agent.json's `route_elite` ({"bonus": floors per elite room, "hp_frac": 0.6 (a whole per cent as a fraction), "act": 1}):
    the Act 1 elite bonus of the route screen (`imitation_chooser.elites_ahead`); None for an agent without it. A bonus that is not a number above 0 (and at
    most `macro_recipe.MAX_ELITE_BONUS`), a fraction that is not a whole per cent from 1 to 100, and an act other than 1 are refused."""
    re_ = meta.get("route_elite")
    if re_ is None:
        return None
    try:
        pct = round(float(re_["hp_frac"]) * 100)
        if abs(float(re_["hp_frac"]) * 100 - pct) > 1e-6:
            raise ValueError(f"hp_frac {re_['hp_frac']!r} is not a whole per cent")
        if re_.get("act", 1) != 1:
            raise ValueError(f"act {re_['act']!r}: the bonus is for Act 1")
        return macro_recipe.elite_bonus(re_["bonus"]), macro_recipe.elite_pct(int(pct))
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{name}: route_elite {re_!r} is not {{'bonus': floors per elite room, 'hp_frac': <whole per cent as a fraction>, 'act': 1}}: {e}") from e


def boss_chain_identity(meta: dict[str, Any], folder: str | Path = "agent.json") -> tuple[Path, tuple[float, float, float]] | None:
    """(the chain file, kappa) of an agent.json's `boss_chain` ({"file": "boss_chain.npz", "sha256": <full>, "kappa": [k1, k2, k3], ...}); None for an agent
    without it. The file must exist in `folder` with the sha256 the block names, kappa is three numbers from 0 to `macro_recipe.MAX_CHAIN_KAPPA`, and the chain
    must have been fitted with this agent's own FightNet (`boss_chain.fight_sha_of(run_value.npz)` equals the chain's `fight_sha256`): the chain reads the same
    fight forecasts the run brain does. Errors name the folder, never a seed."""
    block = meta.get("boss_chain")
    if block is None:
        return None
    folder = Path(folder)
    name = folder.name
    try:
        file = folder / str(block["file"])
        kappa = macro_recipe.chain_kappa(block["kappa"])
        want = str(block["sha256"])
    except (KeyError, TypeError, ValueError) as e:
        raise ValueError(f"{name}/agent.json: boss_chain {block!r} is not {{'file': ..., 'sha256': ..., 'kappa': [k1, k2, k3]}}: {e}") from e
    if not file.is_file():
        raise FileNotFoundError(f"{name}/agent.json: boss_chain names {file.name}, which is not in the folder")
    if file_sha256(file) != want:
        raise ValueError(f"{name}/{file.name} is not the boss chain the agent.json names")
    import boss_chain  # noqa: PLC0415 -- numpy only

    scorer = boss_chain.ChainScorer(file)
    run_value = folder / "run_value.npz"
    if not run_value.is_file() or scorer.fight_sha256 != boss_chain.fight_sha_of(run_value):
        raise ValueError(f"{name}/{file.name} was fitted with another FightNet than this agent's run brain")
    return file, kappa


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
            sims: int, *, kind: int | None = None, grade: int | None = None, grade_sims: int | None = None) -> None:
        """One searched decision. `kind` (a `fight_rows.KIND_NAMES` index), `grade` (the stakes multiplier of the fight's search) and
        `grade_sims` (the simulations the config asked for here) are the plain job's (D65); a row without them is as it always was."""
        self._open.append(len(self.rows))
        row = {"tokens": tokens, "visits": visits, "qvals": qvals, "legal": legal, "encounter": int(encounter),
               "deck": int(deck), "relics": int(relics), "sims": int(sims), "fight": self.fight, "label": 0}
        if kind is not None:
            row.update(kind=int(kind), grade=int(grade), grade_sims=int(grade_sims))
        self.rows.append(row)

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
        graded = [x for x in r if "kind" in x]
        if graded and len(graded) != len(r):
            raise ValueError("a run's rows are all graded or none of them is")
        out = {
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
        if graded:
            out["fight_kind"] = np.array([x["kind"] for x in r], dtype=np.int8)
            out["grade"] = np.array([x["grade"] for x in r], dtype=np.int16)
            out["grade_sims"] = np.array([x["grade_sims"] for x in r], dtype=np.int32)
        return out


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

    def tap(self, kinds, cfg, *, probe: bool = False) -> "FightTap":
        """The tap that keeps this run's searched decisions of the fight `kinds` as graded rows (the plain job's `--combat-rows`)."""
        return FightTap(self, kinds, cfg, probe=probe)

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


class FightTap:
    """What a fight player calls around each of its searches to keep the position as a graded training row (D65, the plain job's
    `--combat-rows`): `wants(pos)` before the search, `encode(root)` the tokens before it (the search must not have touched the
    root), `record(root, res, tokens, pos)` after it with the search's result. It reads the position and the result and nothing else:
    the search, its random draws and the move taken are the ones a run without a tap makes. The rows go to the run recorder's
    `samples` (`rec.samples`, the open fight's encounter, deck and relics from `rec.fight`), graded by the config the fighter searches
    with: the fight's kind, the stakes multiplier of its search and the simulations the config asked for at this decision. The
    recorder's own count of a fight's decisions is not touched, so the run's RunRecord is the same with and without the tap.

    `kinds` are `search_configs.Position.fight_kind` names (`fight_rows.KINDS_OF`). `encode_fn(root, probe)` and `legal_fn(root)` are the
    native readers by default (tests pass stand-ins)."""

    def __init__(self, rec, kinds, cfg, *, probe: bool = False, encode_fn: Callable | None = None, legal_fn: Callable | None = None) -> None:
        self.rec, self.kinds, self.cfg, self.probe = rec, tuple(kinds), cfg, bool(probe)
        self._encode, self._legal = encode_fn, legal_fn
        self.rows: dict[str, int] = {}
        self.seconds = 0.0  # perf_counter time spent in encode and record: process_time ticks at 15.6 ms on Windows, coarser than a call

    def wants(self, pos) -> bool:
        return pos is not None and pos.fight_kind in self.kinds

    def encode(self, root):
        t0 = time.perf_counter()
        try:
            if self._encode is not None:
                return self._encode(root, self.probe)
            from combat_tokens import encode  # noqa: PLC0415

            return encode(root.handle, root.obs, root.encounter_id, probe=self.probe)
        finally:
            self.seconds += time.perf_counter() - t0

    def record(self, root, res, tokens, pos) -> None:
        if len(res.children) < 2:
            return  # a forced move teaches the policy nothing
        t0 = time.perf_counter()
        try:
            if self._legal is not None:
                legal = np.asarray(self._legal(root), dtype=np.int8)
            else:
                from sts2_gym import native  # noqa: PLC0415

                mask = native.valid_actions(root.handle, fight_rows.MAX_ACTIONS)
                legal = np.array([1 if mask[a] else 0 for a in range(fight_rows.MAX_ACTIONS)], dtype=np.int8)
            visits, q = root_targets(res.children, fight_rows.MAX_ACTIONS)
            f = self.rec.fight
            kind = fight_rows.KINDS_OF["all"].index(pos.fight_kind)
            self.rec.samples.add(tokens, visits, q, legal, f["encounter"], f["deck"], f["relics"], res.sims, kind=kind,
                                 grade=self.cfg.stakes_mult(pos), grade_sims=self.cfg.sims_for(pos))
            name = fight_rows.KIND_NAMES[kind]
            self.rows[name] = self.rows.get(name, 0) + 1
        finally:
            self.seconds += time.perf_counter() - t0

    def stats(self) -> dict[str, Any]:
        """The rows kept by kind and the seconds the tap itself took (encoding, the legal mask, the root's targets)."""
        return {"rows": dict(self.rows), "seconds": round(self.seconds, 4)}


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
    for key in ROW_EXTRA_KEYS:  # the plain job's grade columns (D65): all of a part's runs carry them, or the part says nothing it cannot back
        if have and any(key in a for _, a in have):
            if not all(key in a for _, a in have):
                raise ValueError(f"{key!r} is in some runs' rows of a part and not in others'")
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
    (they are replayed) or the part beside them (it is adopted here).

    A flush also happens when the oldest pending run has waited `flush_age_s` (`tick`, which the block's loop calls; None: only full
    parts), so a hard kill loses at most that long's finished runs; `gen0.run_block` flushes at once on a stop request, so a stop the
    lab asks for loses only the runs still in flight."""

    def __init__(self, out_dir: Path, generation: int, *, config: str, base_sims: int, ascension: int, agent: dict,
                 part_runs: int = PART_RUNS, flush_age_s: float | None = FLUSH_AGE_S, clock: Callable[[], float] = time.monotonic) -> None:
        from run_train import shards  # noqa: PLC0415

        self.dir = Path(out_dir) / "shards"
        self.dir.mkdir(parents=True, exist_ok=True)
        self.generation, self.part_runs = generation, part_runs
        self.flush_age_s, self.clock = flush_age_s, clock
        self._oldest: float | None = None  # when the first run now pending was added
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
        if not self.pending:
            self._oldest = self.clock()
        self.pending.append((record["seed"], arrays))
        tag = record.get("prior_path")
        self.pending_paths.append(tag)
        self.prior_paths[tag or "unknown"] = self.prior_paths.get(tag or "unknown", 0) + 1
        self.runs += 1
        if len(self.pending) >= self.part_runs:
            self.flush()
        else:
            self.tick()

    def tick(self) -> bool:
        """Write the pending runs as a part when the oldest has waited `flush_age_s`; whether it did. Cheap when there is nothing to write."""
        if self.flush_age_s is not None and self.pending and self._oldest is not None and self.clock() - self._oldest >= self.flush_age_s:
            self.flush()
            return True
        return False

    def flush(self) -> None:
        self._oldest = None
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
                    device: str | None = None, value_generations: int = GEN_WINDOW, beta: float | None = None) -> dict:
    """Fit FightNet and V_run (DMC target, `run_train.train.fit`) on the window's run records, then the 5-fold out-of-fold fits
    the keep verdict rests on (D33, `keep.cross_fit_keep`). `device` (the train job's) puts the fits there; none: the CPU.
    `value_generations` is V's buffer in generations (the train job's: GEN_WINDOW); `beta` the bootstrap blend (None:
    TrainConfig's 0.3; the fighter-matched fit of `fit_run_value_matched` passes 0, pure Monte Carlo)."""
    import torch  # noqa: PLC0415
    from run_train import keep  # noqa: PLC0415
    from run_train import train as rt  # noqa: PLC0415
    from run_train.engine import CombatBudget  # noqa: PLC0415

    t0, c0 = time.perf_counter(), time.process_time()
    pt = Path(out) / "run_value.pt"
    rt.fit(shard_paths, None, pt, generation=generation, combat_budget=CombatBudget(base_sims, base_sims), ascension=ascension,
           config=rt.TrainConfig(value_generations=value_generations, threads=threads, **({"device": device} if device else {}),
                                 **({"beta": float(beta)} if beta is not None else {})),
           evaluate=keep.evaluate_value_net, cross_fit=keep.cross_fit_keep)
    meta = torch.load(pt, map_location="cpu", weights_only=True)["meta"]
    value = dict(meta.get("value") or {})
    return {"wall_min": (time.perf_counter() - t0) / 60, "cpu_min": (time.process_time() - c0) / 60,
            "fight": meta.get("fight"), "value": value, "config": meta.get("config")}


# ------------------------------------------------------------------ the fighter-matched V_run fit (reset order step C)

VRUN_FIT_JSON = "vrun_fit.json"  # what `fit_run_value_matched` writes beside run_value.pt / .npz: the window, the recipe, the keep record
MATCHED_BETA = 0.0  # pure Monte Carlo: no part of the target is read off the net that played (its p_floor is another fighter's)
MATCHED_RECIPE = ("V_run fit only on the runs played by one combat prior (the candidate's), pure Monte Carlo floor target "
                  "(docs/diagnostics/2026-11-02-recipe.md in the Spireline repo)")


def exploring_manifest(manifest: dict) -> bool:
    """Whether a self-play block's runs were played by an exploring agent (`exploring_agent`): its manifest's agent block names a
    softmax temperature or a uniform share (`Agent.describe` leaves both out for an argmax agent)."""
    agent = manifest.get("agent") or {}
    return bool(agent.get("tau") or agent.get("eps"))


def fighter_manifests(root: Path, prior_sha256: str, gens: Sequence[int] | None = None, *,
                      exploring_only: bool = False) -> list[tuple[Path, dict]]:
    """The self-play shard manifests under `root` whose runs were played by the combat prior `prior_sha256` (the manifest's
    agent block names it), optionally only of the generations `gens`, and with `exploring_only` only the blocks an exploring agent
    played (D47 counts and fits on those: an argmax run carries no outcome for an option the head never ranks first). A manifest
    without a prior sha is never matched."""
    if not prior_sha256:
        return []
    want = None if gens is None else {int(g) for g in gens}
    return [(d, m) for d, m in find_manifests(root)
            if ((m.get("agent") or {}).get("prior_sha256") or None) == prior_sha256
            and (want is None or int(m["generation"]) in want)
            and (not exploring_only or exploring_manifest(m))]


def fit_run_value_matched(shards_root: Path, prior_sha256: str, out: Path, *, gens: Sequence[int] | None = None,
                          beta: float = MATCHED_BETA, threads: int = 4, device: str | None = None,
                          seed_dirs: Sequence[Path | str] = (), seed_dir: Path | None = None,
                          incumbent: Path | str | None = None, ranking_check: Any = None,
                          exploring_only: bool = False, ranking_runs_root: Path | str | None = None,
                          human_leg: bool = True) -> dict[str, Any]:
    """A V_run fit (FightNet, the value head and the D33 keep record) on the runs one combat prior played, into `out`
    (run_value.pt, run_value.npz, vrun_fit.json): the fighter-matched recipe of the reset order's step C. The window is every
    self-play block of that prior (`fighter_manifests`), all of its generations in V's buffer; no E-block seed may be among its
    runs (the train job's rule). `build_candidate(vrun_fit=out)` puts the fit into a candidate when its keep record passes.
    With `exploring_only` the window is the blocks an exploring agent played (D47's rule for generation 12's second attempt); the
    record's `window.exploring_only` says so, and `fit_window_seeds` reads the same window back.

    The keep record also carries the ranking check (D47, D49): the fit's options ranked against `incumbent`'s head (a run_value.npz,
    a generation directory or a current.json) on the fixed set of held-out decisions in the self-play shards under `shards_root`,
    the fit window's runs left out. Since D64 (owner, 2026-11-16) a fit is `kept` when its export exists (`vrun_keep_passed`); the D33 conditions
    and the ranking check are computed and recorded (the keep record says `advisory`, `gated_on` ["export"], and `old_rule_passed`, what the old
    rule would have said) and decide nothing. With no `incumbent` the check cannot run and its block says so.

    `ranking_runs_root` is where the ranking check's fixed set is drawn from (the lab's self-play folder holding the generation-9 pool) when
    the window's root is not that folder (a store of plain runs, D60 item 10); default `shards_root`. `human_leg` False runs the check without
    its human leg (`run_train.ranking.run_check(human=None)`: the shard leg only), for a lane whose rules keep it off the human exam fifth: the
    check then records the human agreement as not run (`human_leg: false`; it gates nothing since D64, so the fit is `kept` all the same)."""
    mans = fighter_manifests(shards_root, prior_sha256, gens, exploring_only=exploring_only)
    if not mans:
        raise ValueError(f"no {'exploring ' if exploring_only else ''}self-play shards of prior {prior_sha256[:12]} under "
                         f"{Path(shards_root).name}")
    window_seeds = [str(x) for d, m in mans for name in m.get("combat_parts", []) for x in read_part(d / name)["seeds"]]
    check_disjoint(window_seeds, known_seed_sets(seed_dir or COMMITTED_SEEDS, None, "", seed_dirs))
    configs = {(str(m.get("config")), int(m["base_sims"]), int(m["ascension"])) for _, m in mans}
    if len(configs) != 1:
        raise ValueError(f"the window mixes search configs, budgets or ascensions: {sorted(configs)}")
    generations = sorted({int(m["generation"]) for _, m in mans})
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    paths = [d / n for d, m in mans for n in m.get("run_shards", [])]
    rv = train_run_value(paths, out, generation=generations[-1], base_sims=int(mans[0][1]["base_sims"]),
                         ascension=int(mans[0][1]["ascension"]), threads=threads, device=device,
                         value_generations=generations[-1] - generations[0] + 1, beta=beta)
    value = dict(rv.get("value") or {})
    keep_doc = value.pop("keep", None) or {"status": "not_run", "all_computable_pass": False}
    if (out / "run_value.npz").is_file():
        check = ranking_check if ranking_check is not None or human_leg else _ranking_without_human_leg
        keep_doc["ranking"] = ranking_block(out / "run_value.npz", incumbent, ranking_runs_root or shards_root, out, exclude_seeds=window_seeds,
                                            check=check)
    keep_advisory(keep_doc)
    doc = {"recipe": MATCHED_RECIPE, "written": utc_now(), "prior_sha256": prior_sha256, "beta": beta, "human_leg": bool(human_leg),
           "window": {"blocks": len(mans), "generations": generations, "exploring_only": bool(exploring_only),
                      "runs_by_generation": {str(g): sum(int(m["runs"]) for _, m in mans if int(m["generation"]) == g)
                                             for g in generations},
                      "jobs": sorted(d.parent.name for d, _ in mans)},
           "run_value": {**rv, "value": value}, "keep": keep_doc,
           "kept": vrun_keep_passed(keep_doc, out),
           "sha256": file_sha256(out / "run_value.npz") if (out / "run_value.npz").is_file() else None}
    write_json(out / VRUN_FIT_JSON, doc)
    return doc


def fit_window_seeds(fit_doc: dict[str, Any], shards_root: Path | str) -> list[str]:
    """The run seeds of the window a `fit_run_value_matched` record (`vrun_fit.json`) was fitted on."""
    mans = fighter_manifests(shards_root, fit_doc["prior_sha256"], fit_doc["window"]["generations"],
                             exploring_only=bool(fit_doc["window"].get("exploring_only")))  # fits before the key was written took every block
    return [str(x) for d, m in mans for name in m.get("combat_parts", []) for x in read_part(d / name)["seeds"]]


def rerank_vrun_fit(fit_dir: Path | str, incumbent: Path | str | None, shards_root: Path | str, *, seed_dir: Path | None = None,
                    seed_dirs: Sequence[Path | str] = (), ranking_check: Any = None) -> dict[str, Any]:
    """Run the ranking check again on a fit `fit_run_value_matched` already wrote (the fit itself is not repeated) and rewrite its
    `vrun_fit.json`: the ranking block and `kept` (the export since D64, so a record written before it that says `kept` false is rewritten
    true). For a check that could not run when the fit was made (the fit takes about
    twenty minutes, the check a few more)."""
    fit_dir = Path(fit_dir)
    doc = read_json(fit_dir / VRUN_FIT_JSON)
    window_seeds = fit_window_seeds(doc, shards_root)
    check_disjoint(window_seeds, known_seed_sets(seed_dir or COMMITTED_SEEDS, None, "", seed_dirs))
    if file_sha256(fit_dir / "run_value.npz") != doc.get("sha256"):
        raise ValueError(f"{fit_dir.name}/run_value.npz is not the fit {VRUN_FIT_JSON} records")
    doc["keep"]["ranking"] = ranking_block(fit_dir / "run_value.npz", incumbent, shards_root, fit_dir, exclude_seeds=window_seeds,
                                           check=ranking_check)
    keep_advisory(doc["keep"])
    doc["kept"] = vrun_keep_passed(doc["keep"], fit_dir)
    doc["reranked"] = utc_now()
    write_json(fit_dir / VRUN_FIT_JSON, doc)
    return doc


def incumbent_head(source: Path | str | None) -> Path | None:
    """The run_value.npz of the head a new fit is ranked against: a .npz itself, a generation directory, or a current.json."""
    if source is None:
        return None
    p = Path(source)
    if p.is_file() and p.suffix == ".json":
        p = Path(read_json(p)["dir"])
    return p / "run_value.npz" if p.is_dir() else p


def ranking_block(candidate_npz: Path, incumbent: Path | str | None, runs_root: Path | str, out_dir: Path, *,
                  exclude_seeds: Sequence[str] = (), check: Any = None) -> dict[str, Any]:
    """The `ranking` block of a keep record (D47, D49): the fit's `run_value.npz` ranked against the incumbent's on the fixed set
    of held-out decisions (`run_train.ranking.run_check`, which writes `ranking_set.json` and `ranking.json` into `out_dir`).
    `check` stands in for `run_check` (tests). A check that cannot run (no incumbent, no head file, an error) is a failing block that
    says why: the fit is not kept on a missing check."""
    from run_train import ranking  # noqa: PLC0415

    inc = incumbent_head(incumbent)
    if inc is None:
        return ranking.not_run("no incumbent head was named to rank the fit against")
    if not inc.is_file():
        return ranking.not_run(f"the incumbent head {inc.parent.name}/{inc.name} is not a file")
    try:
        return (check or ranking.run_check)(candidate_npz, inc, runs_root, out_dir, exclude_seeds=exclude_seeds)
    except Exception as err:  # noqa: BLE001 -- recorded in the block; the fit stays unkept
        return ranking.not_run(f"{type(err).__name__}: {err}")


def _ranking_without_human_leg(candidate_npz: Path | str, incumbent_npz: Path | str, runs_root: Path | str, out_dir: Path | str,
                               **kw: Any) -> dict[str, Any]:
    """The ranking check with its shard leg only (`run_train.ranking.run_check(human=None)`): its record has the human check as missing."""
    from run_train import ranking  # noqa: PLC0415

    return ranking.run_check(candidate_npz, incumbent_npz, runs_root, out_dir, human=None, **kw)


HUMAN_LEG_MISSING = "the human leg did not run"  # run_train.ranking.judge's reason for the human check of a record made without the human leg


def passed_but_human_leg(keep_doc: dict[str, Any] | None) -> bool:
    """Whether a keep record passes everything but the human leg of its ranking check: the D33 conditions, a ranking block that ran (status
    ok) with every check it computed passing (the rest and heal rates on the fixed set), and one failing check, the human agreement, failing only
    because the human leg did not run. A failed human agreement, a failed rate, a check that did not run or a failed D33 condition is never this."""
    block = (keep_doc or {}).get("ranking")
    if not (keep_doc and keep_doc.get("all_computable_pass")) or not block or block.get("status") != "ok":
        return False
    checks = list(block.get("checks") or [])
    human = [c for c in checks if str(c.get("name", "")).startswith("human_agreement")]
    rest = [c for c in checks if not str(c.get("name", "")).startswith("human_agreement")]
    return (bool(rest) and all(c.get("pass") for c in rest) and len(human) == 1 and not human[0].get("pass")
            and human[0].get("why") == HUMAN_LEG_MISSING)


def human_leg_missing(keep_doc: dict[str, Any] | None) -> bool:
    """Whether a keep record's ranking block has its human agreement as not run (the fit's lane kept off the human exam fifth)."""
    block = (keep_doc or {}).get("ranking") or {}
    return any(str(c.get("name", "")).startswith("human_agreement") and c.get("why") == HUMAN_LEG_MISSING for c in block.get("checks") or [])


def read_vrun_fit(fit_dir: Path | str, *, without_human_leg: bool = False) -> dict[str, Any]:
    """A fighter-matched fit's record, checked: the export (`run_value.npz`) is there and is the file the record names. Since D64 that is all:
    nothing in the keep record gates, and a record written before it that says `kept` false (a heal rate or a D33 interval refused it) reads
    as the fit it is. `without_human_leg` is accepted for the callers that still pass it and does nothing."""
    fit_dir = Path(fit_dir)
    doc = read_json(fit_dir / VRUN_FIT_JSON)
    npz = fit_dir / "run_value.npz"
    if not vrun_keep_passed(doc.get("keep"), fit_dir):
        raise ValueError(f"the V_run fit in {fit_dir.name} has no export (run_value.npz); a candidate never carries it")
    if file_sha256(npz) != doc.get("sha256"):
        raise ValueError(f"{fit_dir.name}/run_value.npz is not the fit {VRUN_FIT_JSON} records")
    return doc


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


def ranking_ok(keep_doc: dict[str, Any] | None) -> bool:
    """Whether the keep record's ranking block (D47, D49: how the fit ranks rest, card and map options against the head it would
    replace; `run_train.ranking`) passes its old limits. Since D64 this is a diagnostic: it is computed for the record and gates nothing
    (`vrun_keep_passed`). A record without the block (made before D49, or a synthetic one) has nothing to say and reads true; a record with
    the block reads its `pass`, and every fit made now carries one: a check that could not run is written as a failing block, never left out."""
    block = (keep_doc or {}).get("ranking")
    return True if block is None else bool(block.get("pass"))


KEEP_GATED_ON = ("export",)  # what decides a V_run fit since D64 (2026-11-16): the export, nothing in the keep record


def vrun_keep_passed(keep_doc: dict[str, Any] | None, cand: Path) -> bool:
    """Whether the candidate holds its V_run export (`run_value.npz`): the one thing that decides a fit since D64 (owner, 2026-11-16: "I think
    this is a dumb rule and doesn't make sense hard coding this rule in").

    The keep record still computes and records the D33 out-of-fold Brier skill (`all_computable_pass`, "predicts at all") and the ranking block
    (`ranking_ok`: the heal and rest rates against the head it would replace, the human agreement), each check's `pass` written as before, so a
    memo can read them; none of them gates. `keep_doc` is accepted for the callers that pass it and is not read: `keep_advisory` marks the record
    `advisory` / `gated_on`. Before D64 this was `all_computable_pass and ranking_ok and export`; the judge's verdict on the E-block is the only
    test a candidate with a fit has left to pass."""
    return (Path(cand) / "run_value.npz").is_file()


def keep_advisory(keep_doc: dict[str, Any]) -> dict[str, Any]:
    """Mark a keep record as what it is since D64: `advisory` true, `gated_on` ["export"], and `old_rule_passed`, what the D33 and ranking rule
    would have said (the verdict that used to decide). The checks inside keep their own `pass`. Returns the same dict, marked."""
    keep_doc["advisory"] = True
    keep_doc["gated_on"] = list(KEEP_GATED_ON)
    keep_doc["old_rule_passed"] = bool(keep_doc.get("all_computable_pass")) and ranking_ok(keep_doc)
    return keep_doc


def macro_vrun_on(flag: bool | None = None) -> bool:
    """Whether the macro switch is on: `flag` when given, else on unless SPIRELINE_MACRO_VRUN opts out (`MACRO_OPT_OUT_VALUES`)."""
    if flag is not None:
        return bool(flag)
    return os.environ.get(MACRO_VRUN_FLAG, "").strip().lower() not in MACRO_OPT_OUT_VALUES


def macro_for(keep_doc: dict[str, Any] | None, cand: Path, enabled: bool | None = None) -> str:
    """`vrun` when the candidate holds its V_run export (`vrun_keep_passed`; the keep record is advisory since D64) and the macro switch is on
    (`macro_vrun_on(enabled)`), else `rules`.

    The switch is on by default since D33 (2026-10-25); SPIRELINE_MACRO_VRUN=rules is the opt-out. A candidate that plays
    V_run's macro is judged with its prior and V_run together; that is the decision (VRUN_SWEEP.md 1.5), and the E-block's
    rule (floor, boss-kill guard, F7 audit) is unchanged."""
    return "vrun" if vrun_keep_passed(keep_doc, cand) and macro_vrun_on(enabled) else "rules"


def ranking_summary(keep_doc: dict[str, Any] | None) -> dict[str, Any] | None:
    """What agent.json and train.json keep of a keep record's ranking block: the verdict, the set's digest and each check's figures."""
    block = (keep_doc or {}).get("ranking")
    if block is None:
        return None
    return {"status": block.get("status"), "pass": block.get("pass"), "set_digest": (block.get("set") or {}).get("digest"),
            "checks": [{k: c.get(k) for k in ("name", "candidate", "incumbent", "diff", "limit", "pass")} for c in block.get("checks") or []]}


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
    """Which V_run fit the candidate carries (D33, D64). A new fit that exported (`vrun_keep_passed`: the keep record is advisory since D64) is
    the candidate's `run_value.npz`; a fit that made no export is moved aside (`rejected-fit/`, whatever it left) and the seed fit (the A/B's,
    `train_generation(vrun_seed=)`) takes its place; with neither the candidate carries no kept fit. Returns the manifest entry: source (new / seed / none), kept, sha256 of the carried file, the
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
    entry: dict[str, Any] = {"rule": "a later fit replaces the seed fit when it exported; its keep record is recorded and gates nothing (D64)",
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
                     vrun_seed: Path | str | None = None, candidate_generation: int | None = None,
                     ranking_check: Any = None) -> Path:
    """Train generation `gen`'s candidate (the next generation's, or `candidate_generation`'s) from the last GEN_WINDOW generations' shards. `seed_dirs`
    are further folders of E-blocks (the lab's generated ones) that no training seed may belong to. `vrun_seed` is a V_run fit
    (`run_value.npz`) the candidate carries unless its own new fit passes the keep test (`carry_vrun_fit`, D33). `candidate_generation`
    names the candidate's generation number when it is not gen + 1 (a new policy skipping the numbers between: it is written as
    gen<target>/candidate, its attempt count starts at zero, and train.json / agent.json carry a `lineage` note). A new V_run fit is
    also ranked against the seed fit (`ranking_block`, D47/D49) and carried only when that check passes too; with no seed fit there
    is nothing to rank against and the check does not apply (`ranking.no_incumbent`). `ranking_check` stands in for the check (tests)."""
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
    if vrun and (cand / "run_value.npz").is_file():  # D47/D49: the new fit's ranking against the seed fit it would replace
        from run_train import ranking  # noqa: PLC0415

        seed_ok = vrun_seed is not None and Path(vrun_seed).is_file()
        doc["vrun_keep"]["ranking"] = (ranking_block(cand / "run_value.npz", vrun_seed, shards_root, cand, exclude_seeds=window_seeds,
                                                     check=ranking_check) if seed_ok else ranking.no_incumbent())
    keep_advisory(doc["vrun_keep"])
    passed, enabled = vrun_keep_passed(doc["vrun_keep"], cand), macro_vrun_on()
    fit = doc["vrun_fit"] = carry_vrun_fit(cand, doc["vrun_keep"], vrun_seed) if vrun else {"source": "none", "kept": False, "sha256": None}
    macro = macro_for(doc["vrun_keep"], cand, enabled) if fit["kept"] else "rules"  # a carried fit (new or seed) and its export decide (D64)
    doc["vrun_keep"]["macro_switch"] = {"passed": passed, "enabled": enabled, "carried": fit["source"], "note": MACRO_SWITCH_NOTE}
    doc["vrun_keep"]["note"] = vrun_keep_note(macro)
    lineage = lineage_note(gen, target, macro, fit["source"])
    if lineage:
        doc["lineage"] = lineage
    write_json(cand / "agent.json", {"generation": target, "macro": macro, "trained_from": gen, "written": utc_now(),
                                     **({"lineage": lineage} if lineage else {}),
                                     "mask_version": CANDIDATE_MASK_VERSION, "vrun_fit": {"source": fit["source"], "sha256": fit["sha256"]},
                                     "note": ("V_run's new fit exported; its keep record (D33 skill, the ranking check) is recorded and gates nothing "
                                              "(D64), and the macro switch is on (D33): it picks macro options" if macro == "vrun" and fit["source"] == "new" else
                                              "the seed V_run fit (the A/B's) is carried: the new fit made no export" if macro == "vrun" else
                                              f"V_run's new fit exported; {MACRO_VRUN_FLAG}=rules opts "
                                              "out, so macro stays rules" if passed else
                                              "macro stays rules: V_run made no export")})
    write_json(cand / "train.json", doc)
    shutil_rmtree(cand / "work")
    return cand


def vrun_keep_note(macro: str) -> str:
    """train.json's vrun_keep note: what the macro verdict means for the candidate's E-block."""
    if macro == "vrun":
        return ("V_run's fit exported; its keep record (D33 skill, ranking check) is recorded and gates nothing since D64, and Tier-B regret and "
                "the battery are not computed; the candidate "
                "plays a new combat prior and V_run's macro together, so its E-block judges them together and a gain or a loss "
                "cannot be attributed to one of them")
    return "macro stays rules: the E-block judges the combat prior alone"


def shutil_rmtree(p: Path) -> None:
    import shutil  # noqa: PLC0415

    shutil.rmtree(p, ignore_errors=True)


# ------------------------------------------------------------------ a candidate assembled from parts (D44 step A)

BUILT_KEY = "built"  # agent.json and train.json of a candidate `build_candidate` assembled: nothing was trained, `train` judges it as it is
BUILT_PRIOR_FILES = ("combat_policy.pt", "combat_net.pt")  # the prior export's files a built candidate copies (model.json is written anew)
BUILT_BASE_FILES = ("run_value.npz", "run_value.pt")  # the base generation's V_run fit, copied unchanged
STUDENT_FILE = "run_student.npz"  # a student candidate's weights (`rl/run_student.py`), and the legs it passed the human bar and the teacher on
STUDENT_LEGS_FILE = "run_student_legs.json"


def candidate_dirname(retrain: int = 0, attempt: int = 1, tag: str = "") -> str:
    """The folder a candidate of generation <target> sits in: `candidate` (a first attempt), `candidate-r<n>` (the n-th retrain of a
    trained one) or `candidate-a<n>` (the n-th attempt of a built one: the same target number tried again, beside the earlier
    attempts' folders, which stay as they are)."""
    if attempt > 1 and retrain:
        raise ValueError("an attempt folder is not a retrain folder")
    if tag:
        if attempt > 1 or retrain or not TAG.fullmatch(tag):
            raise ValueError(f"a recipe tag is lowercase words joined by hyphens, never beside an attempt or a retrain: {tag!r}")
        return f"candidate-{tag}"
    if attempt > 1:
        return f"candidate-a{int(attempt)}"
    return "candidate" if retrain == 0 else f"candidate-r{retrain}"


def built_candidate(agent_root: Path | str, target: int, retrain: int = 0, attempt: int = 1, tag: str = "") -> Path | None:
    """AGENT/gen<target>/candidate (candidate-r<n> for a retrain index, candidate-a<n> for an attempt, candidate-<tag> for a tagged
    recipe) when `build_candidate` wrote it (agent.json has `built`), else None: `gen0.py train` judges such a candidate as it is and
    never trains over it."""
    cand = Path(agent_root) / f"gen{int(target)}" / candidate_dirname(retrain, attempt, tag)
    meta = cand / "agent.json"
    return cand if meta.is_file() and BUILT_KEY in read_json(meta) else None


def build_candidate(prior_dir: Path | str, base: Path | str, agent_root: Path | str, target: int, *, why: str,
                    export: bool = True, vrun_fit: Path | str | None = None, attempt: int = 1, tag: str = "",
                    human_prior: Path | str | None = None, lam: float | None = None, pick: str = macro_recipe.PICK_ARGMAX,
                    skip_bias: float = 0.0, search_schedule: dict[str, Any] | None = None, potion_hold: float | None = None,
                    lam_acts: Mapping[int, float] | None = None, student: Path | str | None = None,
                    student_lams: Mapping[str, Sequence[float]] | None = None, student_keep: Sequence[str] | None = None,
                    student_notes: Mapping[str, Any] | None = None, route_elite: tuple[float, float] | None = None) -> Path:
    """Generation `target`'s candidate assembled from parts, nothing trained (D44 step A): the combat prior exported in `prior_dir`
    (model.json, combat_policy.pt and combat_net.pt; the fight loop's `players/v<N>` has that layout) with everything else of the
    generation `base` (a generation directory or a current.json): its V_run fit, macro, tau, eps and mask version.

    Written as AGENT/gen<target>/candidate in a trained candidate's layout (model.json by `write_prior_model_json`, the ONNX files by
    `export_all`, agent.json, train.json), both json files with `built` naming the two sources and their sha256, and a `lineage`
    note when `target` skips numbers. The prior must be one this build plays (`combat_prior.find`: the same token vocabulary, the
    sha its model.json names) and search with the settings every candidate's model.json gives (`PRIOR_SEARCH`, probe off), so the
    candidate plays exactly that network; an existing candidate directory is never written over. `gen0.py train` with a
    `ready-<base generation>-g<target>` marker judges it on the promotion path (`promote`), as a trained candidate is judged.

    `vrun_fit` (a folder `fit_run_value_matched` wrote) replaces the base's V_run fit with that one, which must have passed the
    keep test (`read_vrun_fit`); the base must play the vrun macro. agent.json's `vrun_fit` then names source "fit", the folder,
    its recipe and whether its runs were played by this candidate's prior (`matched`).

    `attempt` (default 1: gen<target>/candidate) places a further attempt at the same generation number in gen<target>/candidate-a<n>
    (`candidate_dirname`), so the earlier attempt's folder, its promotion.json and its markers stay as they are; the lab's train job
    takes it by the `ready-<base>-g<target>-a<n>` marker.

    `human_prior` (the weights of rl/human_prior.py's fit, `weights.npz`) with `lam` makes the blended chooser the candidate's macro
    (`imitation_chooser`, D55 milestone 4): the file is copied as human_prior.npz and agent.json's `macro` is the recipe string
    `vrun+pi_h:lambda=<lam>:pi_h=<8 hex of its sha256>` (`macro_recipe.macro_name`), which the ledger counts attempts under as a
    recipe of its own. The base must play the vrun macro (its V_run fit is the other half of the blend). `tag` places the candidate in
    gen<target>/candidate-<tag> (marker `ready-<base>-g<target>-<tag>`), so two recipes of one generation number sit side by side.

    `pick` and `skip_bias` (with `human_prior`) say how the chooser takes its option (`macro_recipe.PICKS`): the argmax (the default),
    a draw from softmax((V + lambda log pi_h) / 1) ("sample"), or the argmax with `skip_bias` floors added to the skip option ("skipcal");
    the mode is part of the macro string. `lam_acts` ({act number from 1: lambda}, with `human_prior`) gives an act its own lambda: the
    `:act<N>=<lambda>` parts of the macro string, agent.json `human_prior.lambda_acts` ({"1": "0"}); an act whose lambda is `lam` is
    dropped from both (`macro_recipe.act_part`), and without the option nothing is written. `search_schedule` ({"base": a search config
    name, "mult": {"hall", "elite", "boss"}}) gives the candidate a stakes schedule and `potion_hold` (a fraction of max HP, a whole per
    cent) the hallway potion hold, both written to agent.json (`gen_loop.search_identity` reads them back), with `recipe`, the ledger's
    recipe of the whole (`candidate_recipe`).

    `student` (a file `human_prior_infer.write_student` wrote: the run student of `rl/run_student.py`) with `student_lams` ({kind: (lambda in
    Act 1, 2, 3)}) makes the recipe `vrun+pi_s:...` (`macro_recipe.student_name`): the student is copied as run_student.npz and takes over those
    kinds at those lambdas, the human prior's blend decides the deck screens it has not taken over. The base may be a vrun generation (then
    `human_prior` and `lam` name the blend) or a blended one, or a student's, or one with a stakes schedule and a hold: the candidate keeps
    the base's human prior, lambda, lambda by act, pick mode, stakes schedule and hold unless they are given, so it differs from the base in
    the student alone (`screens.kind_of` reads it as `screen:student`).

    `student_keep` (kinds, with `student`; D59 lane G): kinds the base's own student serves that the candidate keeps, with the base's net and the lambdas the
    base's recipe gives them. The candidate's run_student.npz is then a merged student (`human_prior_infer.write_merged_student`): `student` serves the kinds of
    `student_lams`, the base's student the kept ones; the recipe names the merged file's sha256 and every kind, and agent.json's `student` block says where the
    kept kinds came from (`kept`). A kind both named in `student_lams` and kept, or one the base's student does not serve, is refused. `student_notes` (a
    mapping) is added to agent.json's `student` block as it is (the builder's record of the lambdas it floored).

    `route_elite` (bonus, HP fraction; D60 item 9, the elite test): the Act 1 elite bonus of the route screen, written to agent.json (`route_elite`:
    {"bonus", "hp_frac", "act": 1}; `route_elite_identity` reads it back) and to the recipe (`+elite:<bonus>@<per cent>`), so it is a recipe of its own. Only a
    blended or student recipe carries it (a base that has one keeps it when a student is given, as the schedule and the hold are kept)."""
    import shutil  # noqa: PLC0415

    import combat_prior  # noqa: PLC0415

    prior_dir, agent_root, target = Path(prior_dir), Path(agent_root), int(target)
    base_agent = load_agent(base)
    if base_agent.path is None:
        raise ValueError("the base must be a generation directory or a current.json, not generation 0's empty agent")
    student_sha, student_kinds = None, None
    if (student is None) != (not student_lams):
        raise ValueError("a student needs its per-kind lambdas, and lambdas need a student")
    if student_keep and student is None:
        raise ValueError("heads kept from the base's student sit beside a new student: give the student too")
    merged_dir = None
    if student is not None:
        import human_prior_infer  # noqa: PLC0415 -- numpy only

        if not Path(student).is_file():
            raise FileNotFoundError(f"no student weights at {student}")
        fitted = human_prior_infer.student_kinds_of(student)
        student_kinds = {str(k): tuple(float(x) for x in v) for k, v in student_lams.items()}
        if any(k not in fitted for k in student_kinds):
            raise ValueError(f"the student was fitted for {sorted(fitted)}, not for {sorted(set(student_kinds) - set(fitted))}")
        student_sha = file_sha256(student)
        kept_block = None
        if student_keep:
            base_lams = {k: tuple(v) for k, v in base_agent.student_lams}
            if not base_agent.student:
                raise ValueError("the base has no student to keep heads from")
            kept = tuple(dict.fromkeys(str(k) for k in student_keep))
            absent = [k for k in kept if k not in base_lams]
            if absent:
                raise ValueError(f"the base's student does not serve {absent} (it serves {sorted(base_lams)}): nothing to keep")
            both = [k for k in kept if k in student_kinds]
            if both:
                raise ValueError(f"{both} would be served by the new student and by the kept head")
            import tempfile  # noqa: PLC0415

            merged_dir = Path(tempfile.mkdtemp(prefix="merged-student-"))
            merged = merged_dir / STUDENT_FILE
            human_prior_infer.write_merged_student(merged, [(student, tuple(student_kinds)), (base_agent.student, kept)])
            kept_block = {"file": f"{Path(base_agent.path).name}/{Path(base_agent.student).name}", "sha256": file_sha256(base_agent.student),
                          "kinds": {k: [macro_recipe.format_lambda(x) for x in base_lams[k]] for k in kept}}
            student_kinds = {**student_kinds, **{k: base_lams[k] for k in kept}}
            student_sha = file_sha256(merged)
        if human_prior is None and lam is None:  # keep the base's blend on the screens the student does not take over
            if not macro_recipe.is_blended(base_agent.macro):
                raise ValueError("a student recipe keeps the human prior's blend on the other kinds: build on a blended base, or give the "
                                 "human prior's weights and lambda")
            human_prior, lam = base_agent.human_prior, base_agent.lam
            lam_acts = dict(base_agent.lam_acts) if lam_acts is None else lam_acts
            if pick == macro_recipe.PICK_ARGMAX and not skip_bias:
                pick, skip_bias = macro_recipe.parse_pick(base_agent.macro)
        base_meta0 = read_json(Path(base_agent.path) / "agent.json") if (Path(base_agent.path) / "agent.json").is_file() else {}
        if search_schedule is None and base_meta0.get("search_schedule") is not None:  # the search stays the base's: only the student differs
            search_schedule = {"base": base_meta0["search_schedule"]["base"], "mult": base_meta0["search_schedule"]["mult"]}
        if potion_hold is None and base_meta0.get("potion_hold") is not None:
            potion_hold = float(base_meta0["potion_hold"]["hp_frac"])
        if route_elite is None and base_meta0.get("route_elite") is not None:  # the chooser's map bonus stays the base's unless one is given
            kept_re = route_elite_identity(base_meta0, Path(base_agent.path).name)
            route_elite = (kept_re[0], kept_re[1] / 100.0)
    fit_doc = read_vrun_fit(vrun_fit) if vrun_fit is not None else None
    if fit_doc is not None and base_agent.macro != "vrun":
        raise ValueError(f"a V_run fit drives the vrun macro; the base plays {base_agent.macro}")
    base_dir = Path(base_agent.path)
    if target <= base_agent.generation:
        raise ValueError(f"a candidate built on generation {base_agent.generation} cannot be generation {target}")
    champ = combat_prior.find(prior_dir)  # PriorError when this build cannot play it
    if champ is None:
        raise FileNotFoundError(f"no combat prior (model.json and combat_policy.pt) in {prior_dir.name}")
    settings = {"c_puct": champ.c_puct, "prior_min_visits": champ.prior_min_visits}
    if champ.probe or settings != PRIOR_SEARCH:
        raise ValueError(f"{prior_dir.name} searches with probe {champ.probe} and {settings}; every candidate's model.json says "
                         f"probe off and {PRIOR_SEARCH}, so the built candidate would not play the prior as it was measured")
    missing = [n for n in BUILT_PRIOR_FILES if not (prior_dir / n).is_file()]
    if missing:
        raise FileNotFoundError(f"{prior_dir.name} has no {', '.join(missing)}")
    if int(attempt) < 1:
        raise ValueError(f"attempt {attempt}: attempts count from 1")
    macro, chooser = base_agent.macro, None
    if (human_prior is None) != (lam is None):
        raise ValueError("the blended chooser needs both the prior's weights and lambda")
    if human_prior is not None:
        if base_agent.macro != "vrun" and not (student is not None and macro_recipe.is_blended(base_agent.macro)):
            raise ValueError(f"the blended chooser adds the prior to the vrun macro; the base plays {base_agent.macro}")
        if not Path(human_prior).is_file():
            raise FileNotFoundError(f"no human prior weights at {human_prior}")
        chooser = {"file": "human_prior.npz", "sha256": file_sha256(human_prior), "lambda": macro_recipe.format_lambda(float(lam)),
                   "source": f"{Path(human_prior).parent.name}/{Path(human_prior).name}",
                   "guard": {"prob": 0.02, "floors": 1.0, "rule": "an option the prior gives under prob is taken only if V prefers it by more than floors"}}
        if pick not in macro_recipe.PICKS:
            raise ValueError(f"unknown pick mode {pick!r}; one of {', '.join(macro_recipe.PICKS)}")
        if pick != macro_recipe.PICK_ARGMAX:
            chooser["pick"] = pick
        if pick == macro_recipe.PICK_SAMPLE:
            chooser["tau"] = 1.0
        if pick == macro_recipe.PICK_SKIPCAL:
            chooser["skip_bias"] = float(macro_recipe.format_bias(skip_bias))
        if student is not None:
            macro = macro_recipe.student_name(float(lam), chooser["sha256"], student_sha, student_kinds, pick, skip_bias, lam_acts)
        else:
            macro = macro_recipe.macro_name(float(lam), chooser["sha256"], pick, skip_bias, lam_acts)
        kept_acts = macro_recipe.parse_act_lambdas(macro)
        if kept_acts:
            chooser["lambda_acts"] = {str(a): macro_recipe.format_lambda(v) for a, v in sorted(kept_acts.items())}
    elif pick != macro_recipe.PICK_ARGMAX or skip_bias:
        raise ValueError("a pick mode belongs to the blended chooser (--human-prior and --lambda)")
    elif lam_acts:
        raise ValueError("a lambda by act belongs to the blended chooser (--human-prior and --lambda)")
    if (base_agent.stakes is not None or base_agent.hold is not None) and student is None:
        raise ValueError("the base has a stakes schedule or a potion hold; build on a plain generation")
    if base_agent.route_elite is not None and student is None:
        raise ValueError("the base has an elite bonus, which a build without a student would drop; build on a plain generation")
    search_meta: dict[str, Any] = {}
    if search_schedule is not None:
        search_meta["search_schedule"] = {"base": str(search_schedule["base"]), "mult": macro_recipe.stakes_mult(search_schedule["mult"])}
    if potion_hold is not None:
        search_meta["potion_hold"] = {"hp_frac": float(potion_hold), "fights": "hall"}
    stakes_tag, hold_per_cent = search_identity(search_meta, f"gen{target}/candidate")  # refuses what the load would refuse
    if stakes_tag is not None:
        search_meta["search_schedule"]["tag"] = stakes_tag
    elite_pair = None
    if route_elite is not None:
        if not macro_recipe.is_blended(macro):
            raise ValueError(f"the elite bonus rides on the run chooser: the candidate's macro {macro!r} is not a blended one")
        search_meta["route_elite"] = {"bonus": float(route_elite[0]), "hp_frac": float(route_elite[1]), "act": 1,
                                      "rule": "an Act 1 route option's score gains bonus floors per elite room on the best path through it, while HP is at or "
                                              "above hp_frac of max HP"}
        elite_pair = route_elite_identity(search_meta, f"gen{target}/candidate")  # refuses what the load would refuse
    recipe = macro_recipe.full_recipe(macro, stakes_tag, hold_per_cent, elite_pair)
    cand = agent_root / f"gen{target}" / candidate_dirname(0, int(attempt), tag)
    if cand.exists() and any(cand.iterdir()):
        raise FileExistsError(f"gen{target}/{cand.name} exists; a built candidate never writes over one")
    cand.mkdir(parents=True, exist_ok=True)
    for name in BUILT_PRIOR_FILES:
        shutil.copyfile(prior_dir / name, cand / name)
    model = write_prior_model_json(cand, target)
    for name in BUILT_BASE_FILES:
        if (base_dir / name).is_file():
            shutil.copyfile(base_dir / name, cand / name)
    if fit_doc is not None:
        for name in BUILT_BASE_FILES:
            (cand / name).unlink(missing_ok=True)
            if (Path(vrun_fit) / name).is_file():
                shutil.copyfile(Path(vrun_fit) / name, cand / name)
    if chooser is not None:
        shutil.copyfile(human_prior, cand / "human_prior.npz")
    student_block = None
    if student is not None:
        shutil.copyfile(merged if kept_block else student, cand / STUDENT_FILE)
        if merged_dir is not None:
            shutil.rmtree(merged_dir, ignore_errors=True)
        student_block = {"file": STUDENT_FILE, "sha256": student_sha, "source": f"{Path(student).parent.name}/{Path(student).name}",
                         "kinds": {k: [macro_recipe.format_lambda(x) for x in v] for k, v in student_kinds.items()},
                         "fitted_for": list(human_prior_infer.student_kinds_of(cand / STUDENT_FILE)),
                         **({"kept": kept_block} if kept_block else {}), **dict(student_notes or {})}
    npz = cand / "run_value.npz"
    fit = {"source": "base", "sha256": file_sha256(npz)} if npz.is_file() else {"source": "none", "sha256": None}
    if fit_doc is not None:
        fit = {"source": "fit", "sha256": file_sha256(npz), "dir": Path(vrun_fit).as_posix(), "recipe": fit_doc.get("recipe"),
               "prior_sha256": fit_doc.get("prior_sha256"), "matched": fit_doc.get("prior_sha256") == champ.policy_sha256,
               "window": fit_doc.get("window"), "beta": fit_doc.get("beta"), "ranking": ranking_summary(fit_doc.get("keep"))}
    base_meta = read_json(base_dir / "agent.json") if (base_dir / "agent.json").is_file() else {}
    source_train = read_json(prior_dir / "train.json") if (prior_dir / "train.json").is_file() else None
    prior = {"dir": prior_dir.as_posix(), "policy_sha256": model["policy_sha256"], "net_sha256": file_sha256(cand / "combat_net.pt")}
    built = {"why": why, "written": utc_now(), "trained": False, **({"attempt": int(attempt)} if int(attempt) > 1 else {}),
             "rule": "the prior's files and the base's V_run fit are copied unchanged; `gen0.py train` judges the candidate as it is",
             "prior": {**prior, "model_json": read_json(prior_dir / "model.json")},
             "base": {"dir": base_dir.as_posix(), "generation": base_agent.generation, "macro": base_agent.macro,
                      "policy_sha256": base_agent.describe()["prior_sha256"], "run_value_sha256": fit["sha256"]},
             **({"chooser": {**chooser, "macro": macro}} if chooser else {}), **({"student": student_block} if student_block else {})}
    fit_text = (f"the V_run fit of {Path(vrun_fit).name} ({'matched to this prior' if fit['matched'] else 'played by another prior'})"
                if fit_doc is not None else "its V_run fit")
    if fit_doc is not None:
        built["vrun_fit"] = fit
    lineage = None if target == base_agent.generation + 1 else {
        "from_generation": base_agent.generation, "skips": list(range(base_agent.generation + 1, target)),
        "difference": (f"the combat prior of {prior_dir.parent.name}/{prior_dir.name} (built, not trained) with generation "
                       f"{base_agent.generation}'s macro {base_agent.macro} and {fit_text}"
                       + (f", the run chooser blended with the human prior ({macro})" if chooser else "")
                       + (f" and the run student on {', '.join(sorted(student_kinds, key=macro_recipe.STUDENT_KINDS.index))}" if student_block else "")),
        "decision": why}
    doc: dict[str, Any] = {"candidate_generation": target, BUILT_KEY: built, **({"lineage": lineage} if lineage else {}),
                           "prior_train": source_train}
    if export:
        doc["onnx"] = export_all(cand)
    parts = (f", searching its fights under the schedule {stakes_tag}" if stakes_tag else "") + (
        f", holding hallway potions at {hold_per_cent} in 100 of max HP" if hold_per_cent else "") + (
        f", steering its Act 1 routes by {elite_pair[0]:g} floors per elite room while HP is at {elite_pair[1]} in 100 of max or more" if elite_pair else "")
    if student_block:
        note = (f"built, not trained: the combat prior of {prior_dir.name} and generation {base_agent.generation}'s V_run fit, the run chooser "
                f"blended with the human prior and the run student on {', '.join(sorted(student_kinds, key=macro_recipe.STUDENT_KINDS.index))} ({macro}){parts}; "
                f"its E-block judges the student")
    elif chooser:
        note = (f"built, not trained: the combat prior of {prior_dir.name} and generation {base_agent.generation}'s V_run fit, "
                f"the run chooser blended with the human prior ({macro}){parts}; its E-block judges the "
                + ("recipe as a whole" if parts else "chooser alone"))
    elif parts:
        note = (f"built, not trained: the combat prior of {prior_dir.name} with generation {base_agent.generation}'s macro "
                f"({base_agent.macro}) and V_run fit{parts}; its E-block judges the fight search alone")
    elif fit_doc is None:
        note = (f"built, not trained: the combat prior of {prior_dir.name} with generation {base_agent.generation}'s macro "
                f"({base_agent.macro}) and V_run fit; its E-block judges the prior swap alone")
    else:
        note = (f"built, not trained: the combat prior of {prior_dir.name} with generation {base_agent.generation}'s macro "
                f"({base_agent.macro}) and {fit_text}; its E-block judges the prior and the fit together")
    meta = {"generation": target, "macro": macro, "built_on": base_agent.generation, "written": utc_now(),
            **({"lineage": lineage} if lineage else {}), "mask_version": base_agent.mask_version,
            **{k: base_meta[k] for k in ("tau", "eps", "W") if k in base_meta},
            **({"human_prior": chooser} if chooser else {}), **({"student": student_block} if student_block else {}), **search_meta,
            **({"recipe": recipe} if recipe else {}),
            "vrun_fit": fit, BUILT_KEY: {"why": why, "trained": False, "prior": prior, "base": built["base"],
                                         **({"attempt": int(attempt)} if int(attempt) > 1 else {}), **({"tag": tag} if tag else {})},
            "note": note}
    write_json(cand / "train.json", doc)
    write_json(cand / "agent.json", meta)  # last: `built_candidate` reads a complete directory
    got = load_agent(cand)
    if (got.generation, got.macro, got.describe()["prior_sha256"]) != (target, macro, champ.policy_sha256):
        raise ValueError(f"gen{target}/{cand.name} does not load as built: {got.describe()}")
    return cand


def build_route_elite_candidate(base: Path | str, agent_root: Path | str, target: int, *, why: str, bonus: float, hp_frac: float = 0.6, tag: str = "elite-probe",
                                export: bool = True) -> Path:
    """The candidate that is the student generation `base` plus the Act 1 elite bonus and nothing else (D60 item 9, lane I): `build_candidate` with the
    base's own combat prior, V_run fit, human prior, student file (the same bytes, merged or plain) and kind lambdas, so the macro string, the files and the
    search are the base's and only the recipe's `+elite:<bonus>@<per cent>` part and agent.json's `route_elite` are new. The base's `run_student_legs.json` is
    copied beside it. The base must play a student recipe. Returns the candidate's folder (gen<target>/candidate-<tag>)."""
    import shutil  # noqa: PLC0415

    base_agent = load_agent(base)
    if base_agent.path is None or not macro_recipe.is_student(base_agent.macro) or not base_agent.student:
        raise ValueError("the elite bonus is built on a generation that plays a student recipe (vrun+pi_s): the candidate keeps its files as they are")
    base_dir = Path(base_agent.path)
    cand = build_candidate(base_dir, base_dir, agent_root, target, why=why, export=export, tag=tag, student=base_agent.student,
                           student_lams=dict(base_agent.student_lams), route_elite=(float(bonus), float(hp_frac)))
    base_macro, got_macro = base_agent.macro, load_agent(cand).macro
    if got_macro != base_macro:  # the same chooser number for number: a macro that differs would be another candidate
        shutil.rmtree(cand, ignore_errors=True)
        raise ValueError(f"the candidate's macro {got_macro!r} is not the base's {base_macro!r}")
    legs = base_dir / STUDENT_LEGS_FILE
    if legs.is_file():
        shutil.copyfile(legs, cand / STUDENT_LEGS_FILE)
    return cand


# ------------------------------------------------------------------ the V_run swap (D60 item 10, lane J)

VRUN_SWAP_VALUE_FILES = ("run_value.npz", "run_value.pt", "run_value.onnx", "fight_net.pt", "fight_net.onnx")  # screens.VALUE_FILES: the fit's, never the base's
VRUN_SWAP_NOT_COPIED = ("agent.json", "train.json", "promotion.json", "model.json")  # written anew (model.json for the target generation) or the base's own record


def build_vrun_candidate(base: Path | str, vrun_fit: Path | str, agent_root: Path | str, target: int, *, why: str, tag: str = "",
                         attempt: int = 1, export: bool = True, without_human_leg: bool = False) -> Path:
    """Generation `target`'s candidate that is the generation `base` (a generation directory or a current.json) with its V_run fit replaced
    by the one in `vrun_fit` (a folder `fit_run_value_matched` wrote) and nothing else: every other file of the base's folder is copied byte
    for byte (the combat prior and its ONNX, the human prior, the run student and its legs), agent.json keeps the base's macro, recipe, human
    prior, student, stakes schedule, hold and mask version, so the candidate differs from the base in the run-level judgement alone
    (`screens.differing` reads ["value"], no screen applies) and its attempts count under the base's recipe with a new run head. The base's
    chooser must read V (the vrun macro or a blend with it). The fit must hold its export (`read_vrun_fit`; the keep record is advisory since
    D64 and travels in agent.json's `vrun_fit`); `without_human_leg` is accepted and does nothing, and agent.json's `vrun_fit.human_leg` and note
    say when the fit's ranking check ran without its human leg. The V export (run_value.onnx, fight_net.onnx) is made anew from the fit (`export`; the combat
    ONNX stays the base's bytes); without `export` the candidate has none, never the base's old one. Written as gen<target>/candidate-<tag>
    (`candidate_dirname`), never over an existing folder; `gen0.py build-vrun` writes the marker `ready-<base gen>-g<target>-<tag>`."""
    import shutil  # noqa: PLC0415

    agent_root, target = Path(agent_root), int(target)
    base_agent = load_agent(base)
    if base_agent.path is None:
        raise ValueError("the base must be a generation directory or a current.json, not generation 0's empty agent")
    if base_agent.macro != "vrun" and not macro_recipe.is_blended(base_agent.macro):
        raise ValueError(f"the base's chooser ({base_agent.macro}) reads no V: a V_run fit changes nothing it plays")
    if target <= base_agent.generation:
        raise ValueError(f"a candidate built on generation {base_agent.generation} cannot be generation {target}")
    fit_doc = read_vrun_fit(vrun_fit, without_human_leg=without_human_leg)
    human_ran = fit_doc.get("human_leg") is not False and not human_leg_missing(fit_doc.get("keep"))
    base_dir, fit_dir = Path(base_agent.path), Path(vrun_fit)
    cand = agent_root / f"gen{target}" / candidate_dirname(0, int(attempt), tag)
    if cand.exists() and any(cand.iterdir()):
        raise FileExistsError(f"gen{target}/{cand.name} exists; a built candidate never writes over one")
    cand.mkdir(parents=True, exist_ok=True)
    for f in sorted(base_dir.iterdir()):
        if f.is_file() and f.name not in VRUN_SWAP_NOT_COPIED and f.name not in VRUN_SWAP_VALUE_FILES:
            shutil.copyfile(f, cand / f.name)
    for name in BUILT_BASE_FILES:
        shutil.copyfile(fit_dir / name, cand / name)
    model = write_prior_model_json(cand, target) if (base_dir / "model.json").is_file() else None
    onnx = None
    if export:
        onnx = export_all(cand)
        if (base_dir / "combat_net.onnx").is_file():  # the fight network is the base's: its export stays the base's bytes
            shutil.copyfile(base_dir / "combat_net.onnx", cand / "combat_net.onnx")
            onnx["combat_net"] = {**(onnx.get("combat_net") or {}), "kept": "the base's file"}
    base_meta = read_json(base_dir / "agent.json")
    npz = cand / "run_value.npz"
    prior_sha = base_agent.describe()["prior_sha256"]
    human_text = "run" if human_ran else ("not run: the ranking check's human leg (the held-out human fifth) was left out by the fit's lane; "
                                          "its shard leg (rest and heal rates on the fixed set) is recorded beside it")
    fit = {"source": "fit", "sha256": file_sha256(npz), "dir": fit_dir.as_posix(), "recipe": fit_doc.get("recipe"),
           "prior_sha256": fit_doc.get("prior_sha256"), "matched": fit_doc.get("prior_sha256") == prior_sha, "window": fit_doc.get("window"),
           "beta": fit_doc.get("beta"), "ranking": ranking_summary(fit_doc.get("keep")), "human_leg": human_text,
           "d33_pass": (fit_doc.get("keep") or {}).get("all_computable_pass"), "gated_on": list(KEEP_GATED_ON)}
    old_fit = file_sha256(base_dir / "run_value.npz") if (base_dir / "run_value.npz").is_file() else None
    prior = {"dir": base_dir.as_posix(), "policy_sha256": prior_sha,
             "net_sha256": file_sha256(cand / "combat_net.pt") if (cand / "combat_net.pt").is_file() else None}
    base_block = {"dir": base_dir.as_posix(), "generation": base_agent.generation, "macro": base_agent.macro, "policy_sha256": prior_sha,
                  "run_value_sha256": old_fit}
    extra = {**({"attempt": int(attempt)} if int(attempt) > 1 else {}), **({"tag": tag} if tag else {})}
    built = {"why": why, "written": utc_now(), "trained": False, "swap": "vrun",
             "rule": "the base's folder copied unchanged but its V_run fit; `gen0.py train` judges the candidate as it is",
             "prior": prior, "base": base_block, **extra, "vrun_fit": fit}
    lineage = None if target == base_agent.generation + 1 else {
        "from_generation": base_agent.generation, "skips": list(range(base_agent.generation + 1, target)),
        "difference": f"generation {base_agent.generation}'s agent with the V_run fit of {fit_dir.name}", "decision": why}
    note = (f"built, not trained: generation {base_agent.generation}'s agent ({base_agent.recipe() or base_agent.macro}) with the V_run fit of "
            f"{fit_dir.name} ({'matched to this prior' if fit['matched'] else 'played by another prior'}) and nothing else; its E-block judges the "
            "V_run fit alone" + ("" if human_ran else "; the fit's ranking check ran without its human leg"))
    meta = {k: v for k, v in base_meta.items() if k not in ("lineage", "note", BUILT_KEY, "vrun_fit")}
    meta.update(generation=target, built_on=base_agent.generation, written=utc_now(), vrun_fit=fit, **({"lineage": lineage} if lineage else {}))
    meta[BUILT_KEY] = {"why": why, "trained": False, "swap": "vrun", "prior": prior, "base": base_block, **extra}
    meta["note"] = note
    doc = {"candidate_generation": target, BUILT_KEY: built, **({"lineage": lineage} if lineage else {}),
           "base_train": read_json(base_dir / "train.json") if (base_dir / "train.json").is_file() else None,
           **({"model_json": model} if model else {}), **({"onnx": onnx} if onnx is not None else {})}
    write_json(cand / "train.json", doc)
    write_json(cand / "agent.json", meta)  # last: `built_candidate` reads a complete directory
    got = load_agent(cand)
    if (got.generation, got.macro, got.recipe(), got.describe()["prior_sha256"]) != (target, base_agent.macro, base_agent.recipe(), prior_sha):
        raise ValueError(f"gen{target}/{cand.name} does not load as the base with a new V_run fit: {got.describe()}")
    return cand


# ------------------------------------------------------------------ the boss chain (the Act 3 week, lane 3b)

CHAIN_FILE = "boss_chain.npz"
CHAIN_FIT_FILE = "boss_chain.json"  # the chain's `fit.json` (the fit's record), copied beside it


def build_chain_candidate(base: Path | str, chain_dir: Path | str, agent_root: Path | str, target: int, *, why: str, kappa, tag: str = "chain") -> Path:
    """Generation `target`'s candidate that is the generation `base` (a generation directory or a current.json) with a boss chain added and nothing else
    (`rl/boss_chain.py`): every file of the base's folder is copied byte for byte except `VRUN_SWAP_NOT_COPIED` (agent.json, train.json, promotion.json and
    model.json, the last written anew for the target generation), `boss_chain.npz` and `boss_chain.json` (the chain's fit record) come from `chain_dir` (a folder
    `boss_chain.py fit` wrote), and agent.json keeps the base's macro, human prior, student, stakes, hold, elite bonus and mask version, adding the block
    `boss_chain` (file, sha256, kappa, clip, q4, the fit's folder and the smoke test's verdict) and the recipe's `+chain:<8 hex>@<k1>/<k2>/<k3>` part. So the
    candidate differs from the base in the recipe alone (`screens.differing` reads ["settings"], no screen applies). The base must play a blended chooser and
    carry no chain; the chain must have been fitted with the base's own FightNet. Written as gen<target>/candidate-<tag>, never over an existing folder; it
    writes no marker (`gen0.py build-chain` prints the marker's name)."""
    import shutil  # noqa: PLC0415

    import boss_chain  # noqa: PLC0415
    import screens  # noqa: PLC0415

    agent_root, target, chain_dir = Path(agent_root), int(target), Path(chain_dir)
    kappa = macro_recipe.chain_kappa(kappa)
    base_agent = load_agent(base)
    if base_agent.path is None:
        raise ValueError("the base must be a generation directory or a current.json, not generation 0's empty agent")
    if not macro_recipe.is_blended(base_agent.macro):
        raise ValueError(f"the boss chain rides on the blended chooser; the base's ({base_agent.macro}) is not one")
    if base_agent.chain is not None:
        raise ValueError("the base already carries a boss chain; a build never swaps one silently")
    if target <= base_agent.generation:
        raise ValueError(f"a candidate built on generation {base_agent.generation} cannot be generation {target}")
    base_dir = Path(base_agent.path)
    scorer = boss_chain.ChainScorer(chain_dir / CHAIN_FILE)
    if scorer.fight_sha256 != boss_chain.fight_sha_of(base_dir / "run_value.npz"):
        raise ValueError(f"{chain_dir.name}/{CHAIN_FILE} was fitted with another FightNet than the base's run brain")
    fit_doc = read_json(chain_dir / "fit.json")
    report = read_json(chain_dir / "report.json") if (chain_dir / "report.json").is_file() else None
    cand = agent_root / f"gen{target}" / candidate_dirname(0, 1, tag)
    if cand.exists() and any(cand.iterdir()):
        raise FileExistsError(f"gen{target}/{cand.name} exists; a built candidate never writes over one")
    cand.mkdir(parents=True, exist_ok=True)
    try:
        for f in sorted(base_dir.iterdir()):
            if f.is_file() and f.name not in VRUN_SWAP_NOT_COPIED:
                shutil.copyfile(f, cand / f.name)
        shutil.copyfile(chain_dir / CHAIN_FILE, cand / CHAIN_FILE)
        shutil.copyfile(chain_dir / "fit.json", cand / CHAIN_FIT_FILE)
        model = write_prior_model_json(cand, target) if (base_dir / "model.json").is_file() else None
        base_meta = read_json(base_dir / "agent.json")
        sha = file_sha256(cand / CHAIN_FILE)
        prior_sha = base_agent.describe()["prior_sha256"]
        verdict = (report or {}).get("verdict") or {}
        block = {"file": CHAIN_FILE, "sha256": sha, "kappa": list(kappa), "clip": boss_chain.CLIP, "read_gen": boss_chain.READ_GEN, "q4": fit_doc.get("q4"),
                 "fight_sha256": scorer.fight_sha256, "fit_dir": chain_dir.as_posix(), "report_verdict": verdict.get("verdict"), "report_band": verdict.get("band"),
                 "report_door_auc": verdict.get("door_auc")}
        recipe = macro_recipe.full_recipe(base_agent.macro, base_agent.stakes, base_agent.hold, base_agent.route_elite, (sha[:8], kappa))
        base_block = {"dir": base_dir.as_posix(), "generation": base_agent.generation, "macro": base_agent.macro, "policy_sha256": prior_sha,
                      "run_value_sha256": file_sha256(base_dir / "run_value.npz") if (base_dir / "run_value.npz").is_file() else None}
        prior = {"dir": base_dir.as_posix(), "policy_sha256": prior_sha,
                 "net_sha256": file_sha256(cand / "combat_net.pt") if (cand / "combat_net.pt").is_file() else None}
        note = (f"built, not trained: generation {base_agent.generation}'s agent ({base_agent.recipe() or base_agent.macro}) with the boss chain {sha[:8]} "
                f"(kappa {'/'.join(macro_recipe.format_lambda(k) for k in kappa)} floors per nat in Acts 1 to 3) and nothing else; only the chain and the recipe differ, "
                "and its judgement tests the chain alone")
        lineage = None if target == base_agent.generation + 1 else {
            "from_generation": base_agent.generation, "skips": list(range(base_agent.generation + 1, target)),
            "difference": f"generation {base_agent.generation}'s agent with a boss chain", "decision": why}
        meta = {k: v for k, v in base_meta.items() if k not in ("lineage", "note", BUILT_KEY, "boss_chain")}
        meta.update(generation=target, built_on=base_agent.generation, written=utc_now(), boss_chain=block, recipe=recipe, **({"lineage": lineage} if lineage else {}))
        meta[BUILT_KEY] = {"why": why, "trained": False, "swap": "chain", "prior": prior, "base": base_block, "tag": tag}
        meta["note"] = note
        doc = {"candidate_generation": target, BUILT_KEY: {**meta[BUILT_KEY], "written": utc_now(), "rule": "the base's folder copied unchanged; the boss chain and the "
                                                           "recipe's +chain part are new; `gen0.py train` judges the candidate as it is", "boss_chain": block},
               **({"lineage": lineage} if lineage else {}), "base_train": read_json(base_dir / "train.json") if (base_dir / "train.json").is_file() else None,
               **({"model_json": model} if model else {})}
        write_json(cand / "train.json", doc)
        write_json(cand / "agent.json", meta)  # last: `built_candidate` reads a complete directory
        got = load_agent(cand)
        differing = screens.differing(cand, base_dir)
        if (got.generation, got.macro, got.recipe(), got.describe()["prior_sha256"]) != (target, base_agent.macro, recipe, prior_sha) or differing != ["settings"]:
            raise ValueError(f"gen{target}/{cand.name} does not load as the base with a boss chain alone: {got.describe()}, differing {differing}")
    except BaseException:
        shutil.rmtree(cand, ignore_errors=True)
        raise
    return cand


# ------------------------------------------------------------------ lab markers


_MARKER = re.compile(r"(ready|done)-(\d+)(?:-r(\d+))?")
# names the candidate generation (and, with `-a<n>`, the attempt: the candidate sits in gen<target>/candidate-a<n>); not counted as a budget by mark_ready
TAG = re.compile(r"[a-z]{3}[a-z0-9]*(?:-[a-z0-9]+)*")  # a recipe tag, three letters or more at the start (never `a`, `a2`, `b2`): `ready-11-g12-imit` is the candidate in gen12/candidate-imit
_MARKER_TARGET = re.compile(r"(ready|done)-(\d+)-g(\d+)(?:-a(\d+)|-(" + TAG.pattern + r"))?")


def marker_parts(path: Path | str) -> tuple[int, int] | None:
    """(generation, retrain index) of `ready-<gen>` (index 0), `ready-<gen>-r<n>` or `ready-<gen>-g<target>[-a<n>]` (index 0: it is
    written as gen<target>/candidate, or candidate-a<n>); None for anything else. `marker_target` gives the `-g` form's target and
    `marker_attempt` its attempt."""
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


def marker_attempt(path: Path | str) -> int:
    """The attempt a `ready-<gen>-g<target>[-a<n>]` marker names (1 for the form without `-a`, and for every other form)."""
    t = _MARKER_TARGET.fullmatch(Path(path).name)
    return int(t[4]) if t and t[1] == "ready" and t[4] else 1


def marker_tag(path: Path | str) -> str:
    """The recipe tag a `ready-<gen>-g<target>-<tag>` marker names ("" for every other form): the candidate is gen<target>/candidate-<tag>."""
    t = _MARKER_TARGET.fullmatch(Path(path).name)
    return (t[5] or "") if t and t[1] == "ready" else ""


def marker_key(path: Path | str) -> tuple[int, int, int, int, str] | None:
    """Marker order: generation, then the plain and `-r<n>` forms by retrain index, then the `-g<target>` forms by target, attempt and tag."""
    parts = marker_parts(path)
    if parts is None:
        return None
    target = marker_target(path)
    return (parts[0], 0, parts[1], 0, "") if target is None else (parts[0], 1, target, marker_attempt(path), marker_tag(path))


def ready_markers(train_dir: Path) -> list[tuple[int, Path]]:
    """`LAB/train/ready-<gen>[-r<n>|-g<target>[-a<n>|-<tag>]]` markers, lowest generation first, within one: the plain, the `-r<n>`, the
    `-g<target>` (by target, then attempt, then tag)."""
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
    There is no limit on the number of budgets (D65 addendum 2, owner: "Lift the cap"): the review guard's cap of two promotion
    attempts per generation (`promotion_guard.ATTEMPT_CAP`) used to stop the markers after the second rejection and is now only
    recorded with each judgement. `review_guard` is accepted for the callers that pass it and changes nothing here.
    """
    d = Path(lab) / "train"
    issued = sum(1 for p in (d.glob("*") if d.is_dir() else []) if (m := _MARKER.fullmatch(p.name)) and int(m[2]) == gen)
    if issued and rejections(tsv, gen + 1) < issued:
        return None
    del review_guard  # the old cap's switch: nothing is refused for the attempt number any more
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


def keep_rule_of(flag: str | None = None) -> str:
    """The promotion rule: `flag` ("v1", "v2" or "v3") when given, else the SPIRELINE_KEEP_RULE variable, else "v1" (today's floor primary,
    the default until the lab's train job names v2). Any other value is an error and not a silent v1: a rule that decides a promotion
    is never guessed."""
    value = flag if flag is not None else os.environ.get(KEEP_RULE_FLAG, "")
    v = str(value).strip().lower()
    if v == "":
        return "v1"
    if v not in KEEP_RULES:
        raise ValueError(f"unknown keep rule {value!r} (the rules are {', '.join(KEEP_RULES)})")
    return v


def keep_v2_module():
    """tools/lab/keep_v2, imported on demand like `review_guard_module`: the pure rule (docs/rl/KEEP_TEST_V2.md). The rule v1 never imports it."""
    if str(ROOT) not in sys.path:
        sys.path.insert(0, str(ROOT))
    from tools.lab import keep_v2  # noqa: PLC0415

    return keep_v2


def min_gain_of(min_gain: float | None, rule: str) -> float | None:
    """Route A's smallest deep_f gain per 100 runs for a judgement under `rule` (2026-11-19, D68 addendum 3: `--min-gain`, a recorded parameter).
    Under the keep rules (v2, v3) None is `keep_v2.MIN_GAIN` (1.0) and any other value is checked (finite, not negative: `keep_v2.check_min_gain`);
    under v1, which has no such minimum, None (or the default 1.0) gives None and any other value is an error, not a silent no-op."""
    if rule in KEEP_TESTS:
        k2 = keep_v2_module()
        return k2.MIN_GAIN if min_gain is None else k2.check_min_gain(min_gain)
    if min_gain is not None and float(min_gain) != MIN_GAIN_DEFAULT:
        raise ValueError(f"--min-gain {min_gain:g} is the keep rules' (v2, v3); rule {rule} has no minimum gain (name --keep-rule v3)")
    return None


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


def run_head_sha(agent: Agent) -> str:
    """The sha256 of the run head (run_value.npz) an agent chooses its run-level options by; "" for an agent with none (macro rules)."""
    npz = agent.macro_npz if agent.macro == "vrun" or macro_recipe.is_blended(agent.macro) else None
    return file_sha256(npz) if npz and Path(npz).is_file() else ""


def candidate_identity(agent: Agent) -> tuple[str, str]:
    """A candidate's identity in promotions.tsv: (its combat prior's sha, its run head's sha). The prior's sha alone named a candidate
    until generation 12's second attempt, which keeps generation 11's fight player and changes only the run head (D47): by the prior
    alone it would have read as the first attempt again and been judged on the first attempt's refused block."""
    return agent.describe()["prior_sha256"] or "none", run_head_sha(agent)


def candidate_recipe(agent: Agent) -> str:
    """The recipe a candidate's attempts are counted under (promotions.tsv `macro`): "" for the plain macros (rules, vrun), the macro's
    own string for a blended chooser (`macro_recipe.recipe_of`), and for an agent with a stakes schedule or a potion hold the macro,
    then `+stakes:<base>@h<H>e<E>b<B>` and `+hold:<N>` (`macro_recipe.full_recipe`), so a schedule and a hold are recipes of their own.
    A new recipe starts its attempts at 1: generation 12's self-play line (the vrun macro, closed by D55 after two refusals) does not
    count against the imitation line."""
    return agent.recipe()


def row_recipe(row: dict[str, str]) -> str:
    """The recipe a promotions.tsv row was judged under; rows from before the `macro` column read "" (the plain macros)."""
    return row.get("macro", "") or ""


def row_identity(row: dict[str, str]) -> tuple[str, str]:
    """The identity a promotions.tsv row records: `cand_sha` and `run_head_sha`. Rows written before the second column existed read ""
    there, so each old row stands for one attempt of its combat prior with no head named (no migration of the file)."""
    return row.get("cand_sha", ""), row.get("run_head_sha", "")


def attempts_before(rows: list[dict[str, str]], cand_gen: int, cand_sha: str, head_sha: str = "", recipe: str = "") -> int:
    """Other candidates of generation `cand_gen` judged so far under the same recipe (`candidate_recipe`): distinct identities (combat
    prior sha, run head sha; `row_identity`) other than (`cand_sha`, `head_sha`). A row without a prior sha counts as one candidate; a
    row without a head sha is the candidate of its prior alone (a pair with the head ""); a row without a `macro` is the plain recipe."""
    me = (cand_sha, head_sha)
    return len({row_identity(r) for r in rows
                if r.get("cand_gen") == str(cand_gen) and row_recipe(r) == recipe and row_identity(r) != me})


def attempts_spent(tsv: Path | None, cand_gen: int, review_guard: bool | None = None, recipe: str = "", keep_rule: str = "v1") -> int | None:
    """Where the old cap would have stopped a round (D65 addendum 2: recorded, not gating; nothing asks this to refuse any more). With the
    review guard on (`review_guard`, default the SPIRELINE_REVIEW_GUARD variable; the keep rule v2 carried the cap, so it counted whatever
    the flag said) and `cand_gen` already judged at the old cap (distinct candidates in `promotions.tsv` by prior sha and run head sha,
    `attempts_before` with no sha): that count, the number the old cap refused a further candidate at. None otherwise (guard off, or
    attempts left). `train` still asks, to say in one line that the round goes on past it."""
    if keep_rule not in KEEP_TESTS and not review_guard_on(review_guard):
        return None
    n = attempts_before(read_promotions(tsv), cand_gen, "", recipe=recipe)
    return n if n >= review_guard_module().ATTEMPT_CAP else None


def previous_drift_flagged(rows: list[dict[str, str]]) -> bool:
    """Whether the last judged candidate's drift check flagged (the drift column; empty while the guard was off)."""
    return bool(rows) and rows[-1].get("drift", "") in ("flagged", "blocked")


def review_guard_verdict(cand_rows: list[dict], base_rows: list[dict], rows: list[dict[str, str]], *, cand_gen: int, cand_sha: str,
                         block: str, boot: int = BOOT_B, apply_cap: bool = True, head_sha: str = "", recipe: str = "") -> dict[str, Any]:
    """The review's guard (promotion_guard.judge) on two arms, with the lineage, the drift history and the attempt count
    from the promotions.tsv rows written before this judgement (`rows`; this block's own rows do not count as lineage).
    The attempt is recorded against the old cap (D65 addendum 2) and refuses nothing. With `apply_cap` false (`rejudge`: a re-read is not
    a new attempt) the attempt is reported against the cap as within it, as before."""
    pg = review_guard_module()
    kills_c, kills_b, dmg_c, dmg_b = pg.from_rows(cand_rows, base_rows, kill=boss_kill)
    history = [r for r in rows if r.get("seed_block") != block]
    prior = attempts_before(rows, cand_gen, cand_sha, head_sha, recipe)
    cap = pg.ATTEMPT_CAP if apply_cap else max(pg.ATTEMPT_CAP, prior + 1)  # a cap the attempt cannot exceed would not have refused it
    verdict = pg.judge(kills_c, kills_b, dmg_c, dmg_b, lineage=lineage_blocks(history), previous_flagged=previous_drift_flagged(history),
                       prior_attempts=prior, cap=cap, boot=boot)
    verdict["attempt"]["cap"] = pg.ATTEMPT_CAP  # report the real cap ("attempt 3/2"), within it or not
    return verdict


def decide_review(floor: dict, boss: dict, wins: dict, cand_win_rate: float, base_win_rate: float, guard: dict[str, Any],
                  alpha: float = ALPHA) -> dict[str, Any]:
    """D24 with the review's guard (docs/PROMOTION_GUARD.md): the floor primary and the 10% co-primary clauses as `decide`;
    in guard mode the boss kill is the review's McNemar guard at its own alpha (0.05); the boss-damage guard, the drift
    check's block are further not-worse checks. The attempt cap is recorded as a check (`checks["attempt"]`, what the old cap said) and
    gates nothing (`attempt_advisory`; D65 addendum 2). A flagged (not blocked) drift sets `review`."""
    d = decide(floor, boss, wins, cand_win_rate, base_win_rate, alpha)
    checks = dict(d["checks"])
    if d["boss_kill_mode"] == "guard":
        checks["boss_kill"] = not guard["kill"]["fires"]
    checks["boss_damage"] = not guard["damage"]["fires"]
    checks["drift"] = not guard["drift"]["blocks"]
    checks["attempt"] = bool(guard["attempt"]["within_cap"])
    return {**d, "checks": checks, "promote": checks_hold(checks), "rule": "D24+" + guard["rule"], "guard_alpha": guard["alpha"],
            "attempt_advisory": True, "review": bool(guard["review"])}


def incumbent_frontier(agent_root: Path | str, rows: list[dict[str, str]], config: str = "a3-v1", *, block: str | Sequence[str] | None = None,
                       blocks: int = 3) -> dict[str, Any]:
    """The keep test v2's frontier (`keep_v2.frontier_record`) for a block, read from the incumbent's own arms: the promoted lineage's
    player on each of its last `blocks` judged blocks (the candidate's arm where it was promoted, the incumbent's where it stayed:
    the blocks `lineage_blocks` pools for the drift check, a rejudged block once by its last row), the arm being
    `<agent_root>/<gen>/<dir>/arms/<block>-<config>/runs.jsonl`. Never a rejected candidate's rows. The rows of `block` itself are left
    out (a block is judged by a frontier fixed before it was played), and a block whose arm is missing is skipped and named under
    `missing`. B1 when there is no block. The record names each arm as a path under the agent folder, never an absolute one."""
    k2 = keep_v2_module()
    root = Path(agent_root)
    current = "gen0"  # the lineage's player before any promotion, relative to the agent folder
    by_block: dict[Any, tuple[str, str]] = {}
    left_out = {block} if isinstance(block, str) else set(block or ())  # a pooled judgement (keep rule v3) leaves out every block it pools
    for i, r in enumerate([x for x in rows if x.get("seed_block") not in left_out]):
        mine = f"gen{r.get('cand_gen')}/{r.get('cand_dir') or 'candidate'}"
        promoted = r.get("decision") == "promoted"
        by_block[r.get("seed_block") or ("row", i)] = (r.get("seed_block") or "", mine if promoted else current)
        if promoted:
            current = mine
    counts: list[tuple[list[int], int]] = []
    used: list[dict[str, Any]] = []
    missing: list[str] = []
    for name, rel in list(by_block.values())[-blocks:]:
        arm = root / rel / "arms" / f"{name}-{config}"
        if not name or not (arm / "runs.jsonl").is_file():
            missing.append(f"{rel}/arms/{name}-{config}")
            continue
        kills, n = k2.block_counts(arm_rows(arm))
        counts.append((kills, n))
        used.append({"block": name, "arm": rel, "n": n, "kills": kills})
    return {**k2.frontier_record(counts, blocks), "config": config, "used": used, "missing": missing}


def extended_before(rows: list[dict[str, str]], cand_sha: str, head_sha: str = "", recipe: str = "", block: str | None = None) -> bool:
    """Whether the keep rule v3 already extended this candidate (its prior, run head and recipe) on another block: a ledger row of its
    identity with rule v3 and route X (`keep_v2.EXTEND_ROUTE`). The block being judged is left out, so reading the extending block again
    gives the extension again."""
    k2 = keep_v2_module()
    return any(row_identity(r) == (cand_sha, head_sha) and row_recipe(r) == recipe and r.get("rule") == k2.LEDGER_RULE_V3
               and r.get("route") == k2.EXTEND_ROUTE and r.get("seed_block") != block for r in rows)


def pooled_arm_rows(cand: Path, base_dir: Path, config: str, blocks: Sequence[str], taken: Sequence[str] = ()) -> tuple[list[dict], list[dict], list[dict[str, Any]]]:
    """The arms of the blocks a v3 judgement pools with its own (`--pool-block`): for each block the candidate's `arms/<block>-<config>` and
    the base's, only the seeds both played (the pairs), the rows of every block in one list per side, and a record per block (`block`, `pairs`,
    both arms' runs, and each side's kills per boss). A block with no arm, no pair, or a seed that another block (or `taken`) already holds is
    an error: a seed counted twice would be a pair counted twice."""
    k2 = keep_v2_module()
    seen = set(taken)
    cand_rows: list[dict] = []
    base_rows: list[dict] = []
    info: list[dict[str, Any]] = []
    for name in blocks:
        paths = (Path(cand) / "arms" / f"{name}-{config}", Path(base_dir) / "arms" / f"{name}-{config}")
        if not all((p / "runs.jsonl").is_file() for p in paths):
            raise ValueError(f"pooled block {name}: an arm is missing ({', '.join(p.parent.parent.name + '/arms/' + p.name for p in paths if not (p / 'runs.jsonl').is_file())})")
        cr, br = arm_rows(paths[0]), arm_rows(paths[1])
        paired = {r["seed"] for r in cr} & {r["seed"] for r in br}
        if not paired:
            raise ValueError(f"pooled block {name}: no seed is in both arms")
        both = paired & seen
        if both:
            raise ValueError(f"pooled block {name}: {len(both)} seeds are in another block of the judgement")
        seen |= paired
        pc, pb = [r for r in cr if r["seed"] in paired], [r for r in br if r["seed"] in paired]
        cand_rows += pc
        base_rows += pb
        info.append({"block": name, "pairs": len(paired), "cand_runs": len(cr), "base_runs": len(br),
                     "kills": {"cand": k2.block_counts(pc)[0], "base": k2.block_counts(pb)[0]}})
    return cand_rows, base_rows, info


def pooled_record(main: str, main_pairs: int, main_rows: tuple[list[dict], list[dict]], info: list[dict[str, Any]]) -> dict[str, Any]:
    """The `pooled` entry of a promotion.json: every block of the judgement (the judged block first), its pairs and each side's kills per boss
    (B1, B2, B3a, win), and the pairs in all."""
    k2 = keep_v2_module()
    seeds = {r["seed"] for r in main_rows[0]} & {r["seed"] for r in main_rows[1]}
    own_c = [r for r in main_rows[0] if r["seed"] in seeds]
    own_b = [r for r in main_rows[1] if r["seed"] in seeds]
    blocks = [{"block": main, "role": "judged", "pairs": len(seeds), "kills": {"cand": k2.block_counts(own_c)[0], "base": k2.block_counts(own_b)[0]}},
              *[{"block": i["block"], "role": "pooled", "pairs": i["pairs"], "kills": i["kills"]} for i in info]]
    return {"blocks": blocks, "pairs": sum(b["pairs"] for b in blocks), "main_pairs": main_pairs}


def keep_v2_verdict(cand_rows: list[dict], base_rows: list[dict], rows: list[dict[str, str]], *, cand_gen: int, cand_sha: str,
                    block: str, frontier_info: dict[str, Any], boot: int = BOOT_B, apply_cap: bool = True, head_sha: str = "",
                    recipe: str = "", rule: str = "v2", pooled: Sequence[str] = (), final: bool = True,
                    min_gain: float | None = None, s1: Mapping[str, Any] | None = None, fight_swap: bool = True) -> dict[str, Any]:
    """The keep test v2 (`keep_v2.judge`) on two arms at the frontier `frontier_info` (`incumbent_frontier`), with the lineage, the drift
    history and the attempt count from the promotions.tsv `rows` written before this judgement (this block's own rows do not count),
    exactly as `review_guard_verdict` reads them. The attempt is recorded against the old cap and refuses nothing (D65 addendum 2). With
    `apply_cap` false (`rejudge`: a re-read is not a new attempt) the attempt is reported against the cap as within it, as before.

    `rule` "v3" is the keep rule v3 (D66, `keep_v2.judge_v3`): `pooled` names the blocks whose arms are pooled with `block`'s (they are left out of
    the history as `block` is), the candidate counts as extended already when the ledger says so or a block is pooled (`extended_before`), and
    `final` says every seed of the block was played (only a final judgement extends). `min_gain` is route A's smallest deep_f gain per 100 runs
    (None: `keep_v2.MIN_GAIN`); the verdict records it (`min_gain`, `routes.A.min_gain`).

    `s1` (v3 only; route S, D68 addendum 6 outcome 4) is the arrival exam's reading for this pair (`keep_v2.s1_reading`, checked against the pair's ruler
    hash, candidate and champion) and `fight_swap` says the candidate is a fight-only swap: the verdict is then `keep_v2.judge_v3`'s with route S (`route_s`);
    with no reading it is exactly as it was."""
    k2 = keep_v2_module()
    left_out = {block, *pooled}
    history = [r for r in rows if r.get("seed_block") not in left_out]
    prior = attempts_before(rows, cand_gen, cand_sha, head_sha, recipe)
    cap = k2.pg.ATTEMPT_CAP if apply_cap else max(k2.pg.ATTEMPT_CAP, prior + 1)  # a cap the attempt cannot exceed would not have refused it
    extra = {"rule": "v3", "extended": bool(pooled) or extended_before(rows, cand_sha, head_sha, recipe, block), "final": final} if rule == "v3" else {}
    if s1 is not None:
        if rule != "v3":
            raise ValueError("route S is the keep rule v3's (--keep-rule v3)")
        extra.update(s1=s1, fight_swap=fight_swap)
    verdict = k2.judge(cand_rows, base_rows, frontier=int(frontier_info["index"]), frontier_info=frontier_info, lineage=lineage_blocks(history),
                       previous_flagged=previous_drift_flagged(history), prior_attempts=prior, cap=cap, boot=boot,
                       min_gain=k2.MIN_GAIN if min_gain is None else k2.check_min_gain(min_gain), **extra)
    verdict["attempt"]["cap"] = k2.pg.ATTEMPT_CAP  # report the real cap ("attempt 3/2"), within it or not
    return verdict


def decide_v2(keep: dict[str, Any]) -> dict[str, Any]:
    """The decision of the keep test v2 (or v3) as `compare_arms` carries it: `keep_v2.judge`'s checks (a route holds, wins, crash, drift, and the
    attempt, recorded and advisory: `attempt_advisory`; under v3 also the floor collapse, `extend`, and `advisory_checks` naming the drift check
    beside the attempt) under the rule's name. The Act 1 kill, Act 1 damage and D24's Act 1 boss co-primary are not checks here (`act1_review`
    reports them)."""
    out = {"checks": dict(keep["checks"]), "wins_co_primary": bool(keep["wins"]["active"]), "promote": bool(keep["promote"]),
           "attempt_advisory": bool(keep.get("attempt_advisory", True)), "alpha": keep["alpha"], "rule": keep["rule"], "route": keep["route"],
           "frontier": keep["frontier"]["name"], "review": bool(keep["review"])}
    if "extend" in keep:  # the keep rule v3: an extension is neither a promotion nor a refusal, and the drift check is advisory beside the attempt cap
        out.update({"extend": bool(keep["extend"]), "advisory_checks": list(keep["advisory_checks"])})
    if keep.get("min_gain", MIN_GAIN_DEFAULT) != MIN_GAIN_DEFAULT:  # D68 addendum 3: a minimum other than +1.0 is on the decision (the default decision is as it was)
        out["min_gain"] = float(keep["min_gain"])
    return out


def arm_rows(arm_dir: Path) -> list[dict]:
    return [json.loads(x) for x in (Path(arm_dir) / "runs.jsonl").read_text(encoding="utf-8").splitlines() if x.strip()]


def arm_summary(rows: list[dict]) -> dict[str, Any]:
    import gen0  # noqa: PLC0415

    t = gen0.engine_block(rows)
    k = sum(boss_kill(r) for r in rows)
    lo, hi = gen0.wilson(k, len(rows))
    return {**t, "boss_kills": k, "boss_kill_rate": k / len(rows) if rows else None, "boss_kill_lo": lo, "boss_kill_hi": hi}


def compare_arms(cand_rows: list[dict], base_rows: list[dict], alpha: float = ALPHA, boot: int = BOOT_B,
                 guard: dict[str, Any] | None = None, keep: dict[str, Any] | None = None) -> dict[str, Any]:
    """The promotion table's numbers from two arms' rows, paired on seed. With `guard` (the review's verdict,
    `review_guard_verdict`) the decision is `decide_review`'s and the verdict is kept under `review_guard`; without, D24's. With `keep`
    (the keep test v2's verdict, `keep_v2_verdict`) the decision is `decide_v2`'s and the verdict is kept under `keep_v2`; the floor,
    Act 1 boss and wins figures are the same as ever (the ledger's columns)."""
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
    if keep is not None:  # `per_arrival` (each arm's pass rate at the first Act 3 boss per 100 arrivals, with its SE) is also a field of promotion.json itself
        return {**out, "keep_v2": keep, **({"per_arrival": keep["per_arrival"]} if "per_arrival" in keep else {}), "decision": decide_v2(keep)}
    if guard is None:
        return {**out, "decision": decide(floor, boss, wins, cw, bw, alpha)}
    return {**out, "review_guard": guard, "decision": decide_review(floor, boss, wins, cw, bw, guard, alpha)}


# ------------------------------------------------------------------ the funnel (D51): a smoke block, then the block in looks

SEQ_SMOKE = 200  # paired seeds of the smoke block, the first look
SEQ_BATCH = 300  # seeds added between the later looks
SEQ_SMOKE_MEAN = -1.0  # D64: no longer a refusal; the smoke block records `smoke_would_refuse` when its mean paired floor difference is below this ...
SEQ_SMOKE_ALPHA = 0.05  # ... and the one-sided 95% lower bound of the mean is also below 0 (or the 99% upper bound is: the later looks' refusal)
SEQ_REFUSE_ALPHA = ALPHA  # an interim look refuses when the one-sided 99% upper bound of the mean is below 0
SEQ_EARLY_ALPHA = 0.001  # all the alpha early promotions may spend, split equally over the interim looks (Bonferroni)
SEQ_RULE = "D51-funnel"
FUTILITY_MIN_SEEDS = 500  # the futility stop (D55 addendum 18) looks from this many paired seeds on (1,000 in its first landing); `--futility-min-seeds`, round1_config `futility`
SCREEN_ON = True  # the cheap screens before the arms (rl/screens.py); `--no-screen` switches them off
FUTILITY_ON = True  # ... and is on unless `--no-futility` / `futility.enabled` false
ADVISORY_CHECKS = ("attempt",)  # as promotion_guard.ADVISORY_CHECKS (tested equal): checks recorded with their value that decide nothing (D65 addendum 2)


def gating_checks(checks: dict[str, bool], advisory: Sequence[str] | None = None) -> dict[str, bool]:
    """A decision's checks without the advisory ones: `advisory` is the decision's own list (`decision["advisory_checks"]`; the keep rule v3
    adds the drift check to the attempt cap), ADVISORY_CHECKS when the decision names none."""
    skip = ADVISORY_CHECKS if advisory is None else tuple(advisory)
    return {k: bool(v) for k, v in checks.items() if k not in skip}


def checks_hold(checks: dict[str, bool], advisory: Sequence[str] | None = None) -> bool:
    """Whether every gating check of a decision holds (the advisory checks keep their value in the record and are not asked)."""
    return all(gating_checks(checks, advisory).values())


def failed_checks(checks: dict[str, bool], advisory: Sequence[str] | None = None) -> list[str]:
    """The gating checks of a decision that do not hold."""
    return [k for k, v in gating_checks(checks, advisory).items() if not v]


STOP_FAILS = ("refuse", "futility")  # the stops of a block that did not promote: the decision's `sequential` check is false for both


def futility_rule(played: int) -> str:
    """The stopping rule's name in the decision and the ledger: `futility@<seeds played>`."""
    return f"futility@{int(played)}"


def seq_looks(n_block: int, smoke: int = SEQ_SMOKE, batch: int = SEQ_BATCH) -> list[int]:
    """The looks of a block of `n_block` seeds: the smoke block, then `batch` more seeds each time, the last look at `n_block`
    (200, 500, 800, 1100, 1400, 1700, 2000 for 2,000 seeds; a shorter last batch where `batch` does not divide the rest).
    A block no longer than the smoke block has the one look, the last."""
    if smoke < 2 or batch < 1:
        raise ValueError(f"the smoke block needs 2 seeds or more and a batch 1 or more (got {smoke}, {batch})")
    if n_block <= smoke:
        return [int(n_block)]
    looks = [int(smoke)]
    while looks[-1] + batch < n_block:
        looks.append(looks[-1] + batch)
    looks.append(int(n_block))
    return looks


def seq_early_z(n_looks: int, alpha_early: float = SEQ_EARLY_ALPHA) -> float:
    """The z an interim look must reach to promote early: the standard normal quantile at 1 - alpha_early / (n_looks - 2), the
    interim promote looks being every look but the smoke block and the last. Infinite when there is none (under three looks)."""
    k = n_looks - 2
    return NormalDist().inv_cdf(1 - alpha_early / k) if k > 0 else math.inf


def seq_stats(diffs: Sequence[float]) -> dict[str, float]:
    """Seeds, mean, standard error (SD with n - 1 over the root of n) and z = mean / se of the paired floor differences."""
    x = np.asarray(diffs, dtype=np.float64)
    n = int(x.size)
    if n < 2:
        return {"n": n, "mean": float(x.mean()) if n else math.nan, "se": math.inf, "z": 0.0}
    mean, se = float(x.mean()), float(x.std(ddof=1)) / math.sqrt(n)
    z = mean / se if se > 0 else (math.inf if mean > 0 else -math.inf if mean < 0 else 0.0)
    return {"n": n, "mean": mean, "se": se, "z": z}


def seq_call(diffs: Sequence[float], j: int, looks: Sequence[int], *, alpha_refuse: float = SEQ_REFUSE_ALPHA,
             alpha_early: float = SEQ_EARLY_ALPHA, smoke_mean: float = SEQ_SMOKE_MEAN, smoke_alpha: float = SEQ_SMOKE_ALPHA) -> dict[str, Any]:
    """The funnel's floor-side call at look `j` (0-based) of `looks`: `action` is "refuse", "eligible" (clears the early-promote z; the
    caller still needs every other check), "continue" or "final" (the last look: today's decision on all the seeds).

    The funnel (D51). A block of N paired seeds is played as looks n_1 < ... < n_L = N: 200 (the smoke block), then 300 more each
    time, the last at N, so 200, 500, 800, 1100, 1400, 1700, 2000. The seeds are the E-block's own in file order: the smoke
    block is its first tenth, the block is as fresh and as disjoint from every known block as before, and the last look is the
    2,000-seed figure of today. At look j let d_i be the candidate's floor minus the incumbent's on seed i, m their mean over the
    n_j seeds played, s = SD / sqrt(n_j) its standard error and z = m / s.

    Refuse (any look but the first and the last): m + 2.326 s < 0, the upper bound of the one-sided 99% interval of the mean below 0,
    so the primary check cannot be met and no other check can rescue a promotion (all must hold). The smoke look (the first) refuses
    nothing on the floor since D64 (owner, 2026-11-16): it plays on and records its figures and `smoke_would_refuse`, what D51's smoke
    rule (m < -1.0 with the one-sided 95% lower bound m - 1.645 s below 0; the second clause follows from the first, whatever s is) or
    the bound above would have said; `funnel` still refuses a candidate that crashes there (the verdict's own crash guard, an
    engineering fault and not a learning verdict). A refusal is recorded like a block refusal (promotion.json, a promotions.tsv row,
    the attempt spent).

    Promote early (looks 2 to L-1; the smoke block never promotes): z >= z_e = Phi^-1(1 - 0.001 / (L - 2)), 3.54 for the seven
    looks (the one-sided 99.98% lower bound of the mean above 0), and today's decision on the seeds played says promote
    (the floor's bootstrap bound at alpha 0.01, the boss-kill and boss-damage guards, the drift check; the attempt cap is recorded only) and
    the drift check is not flagged. Otherwise play on. At the last look the decision is today's, unchanged, on all N seeds.

    Error rates. (The table was re-run for D64, when the smoke look stopped refusing; the figures are `test_funnel.py` ErrorRateTests'
    own model, 300,000 paths a row.) Promotion happens at the last look by today's test (one-sided alpha 0.01), or earlier by crossing z_e.
    With no improvement (true mean 0 or below) an early crossing has probability at most (L - 2) * 0.001 / (L - 2) = 0.001, a
    Bonferroni split of a tenth of today's alpha over the five interim looks (the looks are positively correlated, so the real
    figure is lower), so P(promote) <= 0.010 + 0.001 = 0.011. Normal model, per-seed SD 7.74 floors (E11's block):

        true mean   today's P(promote)   funnel P(promote)   seeds played, mean
          0.00        0.0100               0.0102              1969
         +0.30        0.277                0.279               1970
         +0.50        0.713                0.714               1836
         +1.07        1.000                1.000                863      (generation 11's block)
         -0.50        0                    0                   1307
         -1.44        0                    0                    510      (generation 10's first block)
         -5.90        0                    0                    500      (generation 12's block; 200 before D64)

    A refusal can only turn a promotion into a refusal: a candidate whose last look would have passed is refused early with
    probability under 0.00001 at every mean of the table (the smoke clause, which added to it, no longer refuses). Why Bonferroni on a small early budget and not
    an O'Brien-Fleming spending: that spending raises the last look's critical value (about 2.36 for 0.01 in all, against
    2.326) and so changes the final test; spending a tenth of alpha early leaves the last look exactly as it is and adds about
    0.0002 to the error under no improvement, at the price of a hard early bar (z 3.54), which only a large improvement clears.
    `tests/python/test_funnel.py` re-runs the table's model."""
    st = seq_stats(diffs)
    n_looks = len(looks)
    z_ref = NormalDist().inv_cdf(1 - alpha_refuse)
    z_e = seq_early_z(n_looks, alpha_early)
    out: dict[str, Any] = {**st, "look": j + 1, "of": n_looks, "upper99": st["mean"] + z_ref * st["se"], "lower99": st["mean"] - z_ref * st["se"],
                           "z_early": z_e}
    if j >= n_looks - 1:
        return {**out, "action": "final", "reason": "every seed played: today's decision"}
    if j == 0:  # D64: the smoke block refuses nothing on the floor; it records what the old rule (and the later looks' bound) would have said
        would = bool((st["mean"] < smoke_mean and st["mean"] - NormalDist().inv_cdf(1 - smoke_alpha) * st["se"] < 0) or out["upper99"] < 0)
        return {**out, "action": "continue", "smoke_would_refuse": would,
                "reason": "smoke block: the floor difference refuses nothing (D64); its figures are recorded"
                          + (f"; the old rule would have refused at mean {st['mean']:+.3f}" if would else "")}
    if out["upper99"] < 0:
        return {**out, "action": "refuse", "reason": f"the 99% upper bound {out['upper99']:+.3f} of the mean paired floor difference is below 0"}
    if j >= 1 and st["z"] >= z_e:
        return {**out, "action": "eligible", "reason": f"z {st['z']:.2f} clears the early-promote z {z_e:.2f}"}
    return {**out, "action": "continue", "reason": "neither bound is reached"}


def futility_check(floors: Sequence[float], deep_diffs: Sequence[float] | None, n_block: int, *, v2: bool, measure: str = "deep_f",
                   boot: int | None = None, rule: str | None = None, extend_possible: bool = False,
                   min_gain: float | None = None) -> dict[str, Any]:
    """The futility stop's reading of the seeds played (`keep_v2.futility_call`): under the keep test v2 both routes (deeper on deep_f, further
    on the floor, each at the keep test's own alpha), under v1 the floor route alone at D24's alpha. `futile` when no route's 95% upper bound
    reaches the gain the route needs to promote on all `n_block` seeds. Under the keep rule v3 (`rule`) route A is the only route, `n_block` is the
    pairs of the whole judgement (the pooled N when a block extends another) and `extend_possible` lowers its need to the gain that earns an
    extension (`keep_v2.futility_routes`). `min_gain` is route A's smallest gain (None: `keep_v2.MIN_GAIN`)."""
    k2 = keep_v2_module()
    b = BOOT_B if boot is None else boot
    gain = {} if min_gain is None else {"min_gain": min_gain}
    if rule == "v3":
        return k2.futility_call(floors, deep_diffs, n_block, routes=("A",), measure=measure, boot=b, extend_possible=extend_possible, **gain)
    if v2:
        return k2.futility_call(floors, deep_diffs, n_block, measure=measure, boot=b, **gain)
    return k2.futility_call(floors, None, n_block, floor_alpha=ALPHA, routes=("F",), boot=b)


def funnel(seeds: Sequence[str], rows_at: Callable[[int], tuple[list[dict], list[dict]]], judge: Callable[[list[dict], list[dict]], dict],
           *, smoke: int = SEQ_SMOKE, batch: int = SEQ_BATCH, keep_rule: str = "v1", frontier: int = 0, futility: int | None = None,
           frontier_name: str = "deep_f", say: Callable[[str], None] | None = None, prefix: tuple[list[dict], list[dict]] | None = None,
           extend_possible: bool = False, min_gain: float | None = None, route_s: bool = False) -> dict[str, Any]:
    """Walk the looks of `seeds` (see `seq_call` for the rule). `rows_at(n)` gives both arms' rows with at least the first n seeds (it
    plays them, or reads recorded rows: a replay is the same walk); `judge(cand_rows, base_rows)` is today's decision on rows
    (`compare_arms` with the guard), asked only at an eligible look. Returns `looks` (one dict per look reached: `seq_call`'s
    numbers and, for an eligible look that did not stop, `held`, the checks that did not hold), `stop` ({n, look, action, reason}:
    action "refuse", "promote" or "final") and both arms' rows on the seeds played. The last look is not judged here.

    With `keep_rule` "v2" each look is `keep_v2.seq_call` on the floor and the deep_f differences at the boss `frontier` (an index,
    0 = B1), and an eligible look promotes only when the route that cleared the early z holds in `judge`'s verdict (`keep_v2`).
    "v3" is the same walk with the keep rule v3's calls (`keep_v2.seq_call_v3`: refuse on a dead route A or a floor collapse, promote early by
    route A alone). `prefix` (both arms' rows, the keep rule v3's pooled blocks) joins every look's pairs: the statistics, the verdicts and
    the futility's whole-judgement N are those of the pooled pairs, the looks still those of `seeds`; the rows returned include it, and a
    look's `n` is the pooled pairs (`own` the block's seeds). `extend_possible` (a first block of the keep rule v3: an extension is still
    available) lets the futility stop ask route A for the gain that earns an extension alone (`keep_v2.futility_routes`). `min_gain` (the keep rules'
    route A, None: `keep_v2.MIN_GAIN`) is the smallest gain every look, the refusal of a dead route A and the futility stop ask for.

    With `route_s` (S2 of route S, the keep rule v3 after S1 held: D68 addendum 6 outcome 4) the looks refuse only (`keep_v2.seq_call_v3(route_s=True)`:
    from 500 pairs on, deep_f's 99% upper bound below 0 or a floor collapse): no look is ever eligible to promote early, there is no futility stop
    (`futility` is not asked) and the last look is `judge`'s, S2's final bound. `route_s` needs the keep rule v3.

    With `futility` (a number of paired seeds; None: off) a look at that many seeds or more that would play on ("continue") also asks
    `futility_check`: when no route's 95% upper bound reaches the gain it needs to promote on the whole block, the walk stops with action
    "futility" (`stop["rule"]` is `futility@<seeds>`) and says which bounds fell short through `say`. Refusals and early promotions come
    first and are as they were; the last look is never a futility look."""
    if route_s and keep_rule != "v3":
        raise ValueError("route S is the keep rule v3's (keep_rule='v3')")
    if route_s:
        futility = None  # S2 has no futility stop
    seeds = list(seeds)
    looks = seq_looks(len(seeds), smoke, batch)
    log: list[dict[str, Any]] = []
    pre_c, pre_b = ({r["seed"]: r for r in rows} for rows in (prefix or ([], [])))
    pre_seeds = sorted(set(pre_c) & set(pre_b))
    pre_rows = ([pre_c[s] for s in pre_seeds], [pre_b[s] for s in pre_seeds])
    n_judged = len(seeds) + len(pre_seeds)  # the pairs of the whole judgement: the block's own and the pooled
    for j, n in enumerate(looks):
        part = set(seeds[:n])
        cand_all, base_all = rows_at(n)
        cand_rows, base_rows = [r for r in cand_all if r["seed"] in part], [r for r in base_all if r["seed"] in part]
        c, b = {r["seed"]: r for r in cand_rows}, {r["seed"]: r for r in base_rows}
        paired = [s for s in seeds[:n] if s in c and s in b]
        if len(paired) != n:
            raise ValueError(f"only {len(paired)} of {n} seeds are paired in both arms at look {j + 1} of {len(looks)}")
        if pre_seeds:
            c, b = {**c, **pre_c}, {**b, **pre_b}
            cand_rows, base_rows = cand_rows + pre_rows[0], base_rows + pre_rows[1]
            paired = pre_seeds + paired
        floors = [float(c[s]["floor"]) - float(b[s]["floor"]) for s in paired]
        v2 = keep_rule in KEEP_TESTS
        deep_diffs = None
        if v2:
            k2 = keep_v2_module()
            deep_diffs = [100.0 * (k2.deep(c[s], frontier) - k2.deep(b[s], frontier)) for s in paired]
            call = k2.seq_call(floors, deep_diffs, j, looks, **({"rule": "v3"} if keep_rule == "v3" else {}),
                               **({} if min_gain is None else {"min_gain": min_gain}), **({"route_s": True} if route_s else {}))
        else:
            call = seq_call(floors, j, looks)
        smoke_crash = None
        if j == 0 and call["action"] == "continue":  # D64: no floor verdict at the smoke look, but a crashing candidate is an engineering fault
            smoke_crash = keep_v2_module().crash_guard(cand_rows, base_rows)  # the verdict's own crash guard, asked on the smoke seeds
            if not smoke_crash["holds"]:
                call = {**call, "action": "refuse",
                        "reason": (f"smoke block: the candidate crashed or lost {smoke_crash['cand']} of {smoke_crash['n']} seeds against the incumbent's "
                                   f"{smoke_crash['base']} (more than {smoke_crash['slack']:g} extra is the crash guard's limit): an engineering fault, "
                                   "not a learning verdict")}
        fut = None
        if futility is not None and call["action"] == "continue" and n >= futility:  # the refusals and the early promotion come first
            fut = futility_check(floors, deep_diffs, n_judged, v2=v2, measure=frontier_name, rule=keep_rule, extend_possible=extend_possible,
                                 min_gain=min_gain)
            if say:
                say(f"futility look at {n} of {len(seeds)} seeds: {fut['text']}" + ("; no route can reach it: the judgement stops" if fut["futile"] else ""))
            if fut["futile"]:
                call = {**call, "action": "futility", "reason": f"{futility_rule(n)}: {fut['text']}"}
        log.append({k: call[k] for k in ("look", "n", "mean", "se", "z", "upper99", "lower99", "action", "reason")
                    + (("deep_mean", "deep_se", "deep_z", "deep_upper95", "routes") if v2 else ())})
        if "smoke_would_refuse" in call:
            log[-1]["smoke_would_refuse"] = call["smoke_would_refuse"]
        if smoke_crash is not None:
            log[-1]["crash"] = smoke_crash
        if fut is not None:
            log[-1]["futility"] = {"futile": fut["futile"], "routes": fut["routes"]}
        stop = {"n": n, "look": j + 1, "action": call["action"], "reason": call["reason"], **({"pooled_pairs": len(paired)} if pre_seeds else {})}
        if call["action"] == "futility":
            stop["rule"] = futility_rule(n)
            stop["futility"] = {"min_seeds": futility, "routes": fut["routes"]}
        if call["action"] == "eligible":
            res = judge(cand_rows, base_rows)
            d = res["decision"]
            flagged = bool((res.get("review_guard") or res.get("keep_v2") or {}).get("review"))
            route_ok = not v2 or bool(set(call["routes"]) & set(res["keep_v2"]["route"]))  # the route that cleared the early z holds on these seeds
            if d["promote"] and not flagged and route_ok:
                stop["action"] = "promote"
                stop["reason"] += "; every check holds on the seeds played"
            else:
                held = (failed_checks(d["checks"], d.get("advisory_checks")) + (["drift flagged"] if flagged else [])
                        + ([] if route_ok else [f"route {''.join(call['routes'])}"]))
                log[-1]["held"] = held
                log[-1]["reason"] += f"; played on: {', '.join(held)} did not hold"
                continue
        if stop["action"] in ("refuse", "promote", "final", "futility"):
            log[-1]["action"] = stop["action"]
            return {"looks": log, "stop": stop, "cand_rows": cand_rows, "base_rows": base_rows}
    raise AssertionError("the last look always stops")  # pragma: no cover


# ------------------------------------------------------------------ the E-block and the promotion run


TSV_COLUMNS = ("written", "cand_gen", "base_gen", "seed_block", "seeds", "d_floor", "floor_lower", "p_floor", "boss_cand",
               "boss_base", "p_boss", "wins_cand", "wins_base", "decision", "cand_sha", "cand_dir",
               # the review's guard: the attempt of the generation (always), the drift verdict and the guard line (empty / "off" when off)
               "attempt", "drift", "guard",
               # the second half of the candidate's identity (`candidate_identity`); empty in rows written before it existed
               "run_head_sha",
               # the run chooser's recipe when it is a blended one (`candidate_recipe`); empty for the plain macros and in older rows
               "macro",
               # the keep test v2 (tools/lab/keep_v2.py `ledger_cells`); empty in every row judged by v1, which is how an older row reads
               "rule", "frontier", "route", "k2_cand", "k2_base", "d_deep", "deep_lower", "p_deep", "floor_lower95")


def guard_cells(guard: dict[str, Any] | None, attempt: int) -> dict[str, Any]:
    """The three guard columns of a promotions.tsv row."""
    if guard is None:
        return {"attempt": attempt, "drift": "", "guard": "off"}
    return {"attempt": attempt, "drift": guard["drift"]["verdict"], "guard": review_guard_module().summary(guard)}


def keep_cells(keep: dict[str, Any], attempt: int) -> dict[str, Any]:
    """The guard columns of a promotions.tsv row judged by the keep test v2, and its nine own columns (`keep_v2.ledger_cells`)."""
    k2 = keep_v2_module()
    return {"attempt": attempt, "drift": keep["drift"]["verdict"], "guard": k2.summary(keep), **k2.ledger_cells(keep)}


def human_bar_note(cand_agent: Agent, base_agent: Agent | None = None) -> dict[str, Any]:
    """The keep test v2's report-only line on the fight exam's human stratum (D55 addendum 11): the judged player's fight network (the
    sha256 of its combat prior; the lab's candidates share one) as the gym player that holds the same bytes, and that player's latest
    reading against the human from the exam's games, or "no human-stratum games for ...". Nothing is played; no rule reads it, and a
    failure to read yields a line saying so, never an error in the verdict. The base agent's network is named too when it is another."""
    try:
        import fight_exam  # noqa: PLC0415 -- pure Python (the exam's readings)

        sha = cand_agent.describe()["prior_sha256"]
        note = fight_exam.lab_human_bar(sha)
        base_sha = base_agent.describe()["prior_sha256"] if base_agent is not None else None
        if base_sha and base_sha != sha:
            note["line"] += " | base " + fight_exam.lab_human_bar(base_sha)["line"].removeprefix("human bar: ")
        return note
    except Exception as error:  # noqa: BLE001
        return {"line": f"human bar: not read ({type(error).__name__})"}


def row_rule(row: dict[str, str]) -> str:
    """The keep rule a promotions.tsv row was judged under: "v2" or "v3" where the `rule` column says so, "v1" in every other row (older rows
    have no such column)."""
    rule = row.get("rule") or ""
    return rule if rule in KEEP_TESTS else "v1"


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


def block_used_by_another(tsv: Path | None, block: str, cand_sha: str, head_sha: str = "", recipe: str = "") -> bool:
    """Whether `block` already judged a candidate other than (`cand_sha`, `head_sha`, `recipe`): another combat prior, the same prior
    with another run head, or the same pair under another chooser recipe (a row without a sha counts as another one; a row without a
    head sha is its prior alone, `row_identity`; a row without a `macro` is the plain recipe)."""
    return any(r.get("seed_block") == block and (row_identity(r), row_recipe(r)) != ((cand_sha, head_sha), recipe)
               for r in read_promotions(tsv))


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
            server_opts=None, resume: bool = False, start_ix: int = 0) -> list[dict]:
    """Play the seeds with `agent`, or reuse the arm if the same seeds and config were already played there. The arm uses a
    prior server only when `server_opts` (a `prior_launch.ServerOpts`) says so; the promotion arms default to the CPU path.

    With `resume` (the funnel's looks: an arm grows from a first part of the block to the whole of it) the seeds asked for are a
    part of what the folder may hold: a finished arm of the same agent with a row for every seed asked is reused (its rows for those
    seeds returned), and otherwise the arm runs into the same folder keeping the rows it has, so only the other seeds are played.
    summary.json then covers every row, with the wall time summed over the looks and the timed-out seeds of all of them.
    `start_ix` is the index of the first seed in its block (the agent's random numbers are [agent seed, index], so a run of seeds taken from the middle of a
    block keeps its place): 0 for a block played from its start, as always."""
    import gen0  # noqa: PLC0415

    arm_dir = Path(arm_dir)
    s = arm_dir / "summary.json"
    want = set(seeds)
    carry: dict[str, Any] | None = None
    if s.is_file():
        old = read_json(s)
        rows = arm_rows(arm_dir)
        wanted = "server" if (server_opts is not None and server_opts.mode != "off") else "cpu"
        played = "server" if old.get("prior_server") else "cpu"  # both arms of one comparison play on one transport
        have = {r["seed"] for r in rows}
        if old.get("config") == config and old.get("agent") == agent.describe() and played == wanted:
            if (want <= have) if resume else (have == want):
                return [r for r in rows if r["seed"] in want] if resume else rows
            carry = old
    kw = {"resume": True} if resume else {}  # a plain arm is played as it always was
    if start_ix:
        kw["start_ix"] = int(start_ix)
    if server_opts is not None and server_opts.mode != "off":
        gen0.run_block_served(seeds, config, procs, arm_dir, opts=server_opts, ascension=ascension, agent=agent, **kw)
    else:
        gen0.run_block(seeds, config, procs, arm_dir, ascension=ascension, agent=agent, **kw)
    rows = arm_rows(arm_dir)
    if not resume:
        return rows
    if carry is not None:  # the looks before this one: their wall time and timed-out seeds stay in the summary
        new = read_json(s)
        new["wall_s"] = round(float(carry.get("wall_s") or 0) + float(new.get("wall_s") or 0), 1)
        merged: dict[Any, dict] = {}
        for x in [*(carry.get("timed_out") or []), *(new.get("timed_out") or [])]:
            merged.setdefault(x.get("seed"), x)
        if merged:
            new["timed_out"] = list(merged.values())
        write_json(s, new)
    return [r for r in rows if r["seed"] in want]


def screens_module():
    import screens  # noqa: PLC0415 -- rl/ is on sys.path whenever gen_loop is imported

    return screens


def fight_exam_play(cand: Path, label: str, procs: int) -> int:
    """`fight_exam.py play` of the candidate's folder on the human stratum alone (`--stratum human`, games to exam/games-<label>.jsonl, resumable),
    under the judgement's own `lab-train` QUIET.lock line (`--allow-quiet`: it does not wait for the line that announces it). The exit code."""
    cmd = [sys.executable, str(ROOT / "rl" / "fight_exam.py"), "play", "--player-dir", str(cand), "--label", label, "--stratum", "human",
           "--allow-quiet", "lab-train", "--no-baseline", "--procs", str(int(procs))]
    return subprocess.run(cmd, check=False).returncode


def run_screen(cand: Path, base_dir: Path, cand_agent: Agent, base_agent: Agent, *, procs: int = 16, say: Callable[[str], None] | None = None) -> dict | None:
    """The screen that applies to this candidate against this incumbent (`screens.kind_of`), run, and its figures: None when none applies; the
    reading (`rejected`, `line`, the figures) or `{"skipped": why}`; `{"error": <name>}` when the screen itself failed (it never fails the judgement)."""
    say = say or (lambda m: print(m, flush=True))
    sc = screens_module()
    kind = sc.kind_of(cand, base_dir)
    if kind is None:
        return None
    try:
        if kind == "student":
            got = sc.run_student(cand, base_dir)
        elif kind == "fight":
            sha, base_sha = candidate_identity(cand_agent)[0], base_agent.describe().get("prior_sha256")
            label = f"lab-g{cand_agent.generation}-{cand.name}-{sha[:8]}"
            if not base_sha:
                got = {"rule": sc.SCREEN_FIGHT, "kind": "fight", "skipped": "the incumbent has no fight network"}
            else:
                got = sc.run_fight(cand, base_sha, label, lambda lab: fight_exam_play(cand, lab, procs))
        else:
            got = sc.run_prior(cand, base_dir)
    except Exception as error:  # noqa: BLE001 -- a screen is a convenience: its failure plays the arms
        got = {"rule": {"fight": sc.SCREEN_FIGHT, "student": sc.SCREEN_STUDENT}.get(kind, sc.SCREEN_PRIOR), "kind": kind, "error": type(error).__name__}
        say(f"{got['rule']}: failed ({type(error).__name__}: {str(error)[:200]}); the arms are played")
        return got
    if got.get("skipped"):
        say(f"{got['rule']}: not applied ({got['skipped']}); the arms are played")
    elif got.get("line"):
        say(got["line"])
    return got


def screen_rejection(cand: Path, cand_agent: Agent, base_agent: Agent, screen_doc: dict, *, name: str, config: str, procs: int, agent_root: Path,
                     attempt: int, cand_sha: str, head_sha: str, recipe: str, rule: str | None, lab: Path | None) -> dict:
    """A candidate a screen rejected: promotion.json (the decision, `screen` with the figures; no arm was played) and a ledger row with seeds 0 and the
    screen's rule in the `rule` column. The attempt is spent as for any rejection; nothing is promoted and no score marker is written."""
    del lab
    srule = screen_doc["rule"]
    doc = {"written": utc_now(), "seed_block": name, "n_seeds": 0, "config": config, "procs": procs, "alpha": ALPHA,
           "cand": {"agent": cand_agent.describe()}, "base": {"agent": base_agent.describe()},
           "decision": {"promote": False, "rule": srule, "checks": {"screen": False}}, "decision_text": "rejected", "cand_sha": cand_sha,
           "run_head_sha": head_sha, **({"macro": recipe} if recipe else {}), **({"keep_rule": rule} if rule else {}), "screen": screen_doc,
           "f7": "not run", "matched_wall_clock": False}
    write_json(cand / "promotion.json", doc)
    append_promotion(Path(agent_root) / "promotions.tsv",
                     {"written": doc["written"], "cand_gen": cand_agent.generation, "base_gen": base_agent.generation, "seed_block": "", "seeds": 0,
                      "decision": "rejected", "cand_sha": cand_sha, "cand_dir": cand.name, "run_head_sha": head_sha, "macro": recipe, "attempt": attempt,
                      "drift": "", "guard": str(screen_doc.get("line", srule)).replace("\t", " "), "rule": screen_doc.get("ledger_rule", srule)})
    return doc


# ------------------------------------------------------------------ route S (D68 addendum 6 outcome 4): the S1 stage and the base-arm reuse check
#
# Route S (docs/rl/KEEP_TEST_V3.md "Route S") judges a fight-only swap where the fight brain acts. S1 is the arrival exam (`rl/act3_starts.py`): the
# candidate's fight network against the champion's from the champion's own first-Act-3-boss arrivals, run as the first stage of `promote` under the lab's
# train job; S2 is the funnel over the whole-run pairs with its looks refusing only (`funnel(route_s=True)`). The flag is `--route-s {auto,off}`, off by default.

ROUTE_S_MODES = ("auto", "off")
FIGHT_SWAP = "fight"  # `built.swap` in the agent.json `rl/gym_wiring.py` writes for a candidate that is the champion with only its fight network replaced
S1_FILE = "s1.json"  # the candidate's S1 reading, beside its promotion.json
S1_EXAM_DIR = "s1"  # the exam's report.txt and results.json for the candidate (a folder of the candidate's)
S1_LEDGER_FILE = "s1-ledger.json"  # beside the ruler: the tests played against it, so no test's draws are used twice
S1_LEDGER_RULE = "s1-ledger-v1"
S1_PROCS = 12  # processes of the ruler's replays and of the exam
LAB_TRAIN_TAG = "lab-train"  # the QUIET.lock tag the lab's runner holds while the train job runs (tools/lab/runner.py TRAIN_TAG): the exam is told to ignore it
BASE_REUSE_SEEDS = 20  # seeds of the champion's stored arm replayed before S2 may reuse the arm
OUTCOME_FIELDS = ("ix", "win", "floor", "end", "crash", "lost", "nodes", "cards", "rests", "boss_damage_acts")  # a replay is the same when these rows are


class S1Blocked(RuntimeError):
    """The S1 stage was stopped or blocked (the night window, PAUSE, HALT, a landing, another's QUIET line, a stop file) or ended incomplete. What is on disk is
    kept: the job exits non-zero, the runner tries it again, and the ruler's replays and the exam's games resume where they stopped."""


class S1Unavailable(RuntimeError):
    """S1 cannot be played for this pair (the champion has no arms to build a ruler from, or the candidate holds the champion's own fight network): the judgement
    goes on under v3 and records why."""


def route_s_note(mode: Mapping[str, Any] | None, s1: Mapping[str, Any] | None) -> str | None:
    """The words a ledger row's `guard` cell carries when route S was asked (`--route-s auto`) and there was no S1 reading to print: `S1 unavailable: <reason>;
    v3 decides` (the reason is `route_s_mode.reason`: not a fight-only swap, the champion has no arm to build a ruler from, the same network twice, no reading
    to read). None when route S was not asked, when it applies, or when S1 was read: keep_v2's own parts then say how it went (holds, does not hold, not applied)."""
    if not mode or mode.get("applies") or s1 is not None:
        return None
    return f"S1 unavailable: {mode.get('reason') or mode.get('why') or 'no reading'}; v3 decides"


def route_s_of(value: str | None) -> str:
    """`--route-s`: "auto" or "off" (the default). Any other value is an error, not a silent off: a rule that decides a promotion is never guessed."""
    v = "off" if value is None or str(value).strip() == "" else str(value).strip().lower()
    if v not in ROUTE_S_MODES:
        raise ValueError(f"unknown --route-s {value!r} (the modes are {', '.join(ROUTE_S_MODES)})")
    return v


def fight_only_swap(cand: Path | str) -> bool:
    """Whether the candidate is a fight-only swap: the `built.swap` field of its agent.json reads "fight", which `rl/gym_wiring.py` writes (and nothing else
    writes: the prior, V_run and chain builders write other values or none) when it builds the champion with only its fight network replaced."""
    try:
        meta = read_json(Path(cand) / "agent.json")
    except (OSError, ValueError):
        return False
    built = meta.get(BUILT_KEY) if isinstance(meta, dict) else None
    return isinstance(built, dict) and built.get("swap") == FIGHT_SWAP


def s1_ruler_dir(base_dir: Path | str, root: Path | str | None = None) -> Path:
    """The champion's ruler folder: `<root>/<generation folder>-<agent folder>` (`rl/act3_starts.py arrivals`' own default, gym-loop/arrival/gen20-candidate-chain)."""
    import act3_starts as a3  # noqa: PLC0415

    base_dir = Path(base_dir)
    return Path(root if root is not None else a3.ARRIVAL_ROOT) / f"{base_dir.parent.name}-{base_dir.name}"


def usable_ruler(ruler: Path, base_dir: Path) -> dict[str, Any] | None:
    """The meta of the ruler in `ruler` when it is this champion's and its starts file hashes to the meta's (`act3_starts.load_ruler`), else None."""
    import act3_starts as a3  # noqa: PLC0415

    try:
        meta, _lite = a3.load_ruler(ruler)
    except (OSError, ValueError, KeyError):
        return None
    ok = meta.get("champion") == f"{base_dir.parent.name}-{base_dir.name}" and meta.get("champion_dir") == f"{base_dir.parent.name}/{base_dir.name}"
    return meta if ok and meta.get("ruler_hash") else None


def sync_ruler_exclusions(ruler: Path, say: Callable[[str], None] | None = None, *, strict: bool = True) -> dict[str, Any] | None:
    """Put the ruler's runs and starts on the practice's exclusion list (`act3_starts.merge_exclusions`: the fight loop's `exam/exclude_runs.json` beside the
    ruler's `arrival/` folder), so the fight loop never practises on the fights the exam plays. Called wherever a stage reads a usable ruler, so a ruler built or
    extended in the train job, and one built outside it (by hand, `judge-design/S1-v5.sh`), reach the practice with no hand step; a ruler already merged changes
    nothing. `strict` (the train job's S1 stage, before the exam plays): a list that cannot be merged (a file that is not the rule's own, one unreadable) is an
    error. Not strict (a reading read from a file, a rejudgement): it is said and the judgement goes on. A loop with no exclusion file is said, never made."""
    import act3_starts as a3  # noqa: PLC0415

    say = say or (lambda m: print(m, flush=True))
    try:
        rec = a3.merge_exclusions(ruler)
    except (OSError, ValueError, KeyError) as why:
        if strict:
            raise
        say(f"S1: the ruler's runs and starts could not be put on the practice's exclusion list ({why})")
        return None
    if rec["wrote"]:
        say(f"S1: the ruler's runs and starts are on the practice's exclusion list now ({rec['added_runs']} runs and {rec['added_starts']} starts added; "
            f"the list before is kept once as {a3.EXCLUDE_BACKUP})")
    elif rec["status"] == "no exclusion file":
        say(f"S1: no exclusion file beside {Path(ruler).parent.name}/: the ruler's runs and starts are not on any practice list")
    return rec


def s1_expect(cand_agent: Agent, base_agent: Agent, ruler_hash: str | None) -> dict[str, Any]:
    """What a reading must name to be applied to this pair: the ruler's hash (None when the ruler cannot be read) and the sha256 of each fight network."""
    return {"ruler_hash": ruler_hash, "candidate": cand_agent.describe().get("prior_sha256"), "champion": base_agent.describe().get("prior_sha256")}


def s1_ledger(ruler: Path) -> dict[str, Any]:
    """The ledger of the tests played against a ruler (`s1-ledger.json`): `tests` is a list of {test, candidate, ...}; a missing or unreadable file is empty."""
    try:
        doc = read_json(Path(ruler) / S1_LEDGER_FILE)
    except (OSError, ValueError):
        doc = None
    if not isinstance(doc, dict) or not isinstance(doc.get("tests"), list):
        doc = {"rule": S1_LEDGER_RULE, "tests": []}
    return doc


def ruler_hash_of(ruler: Path) -> str | None:
    """The hash in a ruler's meta.json (what its tests are played on); None when there is no readable meta."""
    try:
        h = read_json(Path(ruler) / "meta.json").get("ruler_hash")
    except (OSError, ValueError, AttributeError):
        return None
    return str(h) if h else None


def s1_used_tests(ruler: Path, ledger: Mapping[str, Any]) -> set[int]:
    """The test indices that have been played against the ruler **at its current hash**: the ledger's entries for this hash (an entry that names no hash counts:
    it cannot be told apart, and a draw is never used twice), and every one whose draws (6k to 6k + 5) a games file in the ruler's `exam` folder holds for this
    hash (a test played by hand, before there was a ledger, still counts). A ruler that grew has a new hash and its tests start at 0: what was played on the
    old hash stays in the ledger and in `exam/<old hash first 8>/` as a record, and is not counted (`act3_starts.retire_exam`, `exam_games`). A ruler whose
    meta cannot be read counts everything."""
    import act3_starts as a3  # noqa: PLC0415

    cur = ruler_hash_of(ruler)
    used = {int(t["test"]) for t in ledger["tests"] if cur is None or t.get("ruler_hash") in (None, cur)}
    for f in a3.exam_games(Path(ruler) / "exam", cur):
        for line in f.read_text(encoding="utf-8").splitlines():
            if line.strip():
                with contextlib.suppress(ValueError, KeyError, TypeError):
                    used.add(int(json.loads(line)["j"]) // a3.ARRIVAL_DRAWS)
    return used


def register_s1_test(ruler: Path | str, reading: Mapping[str, Any], cand_dir_name: str = "") -> bool:
    """Record a test played elsewhere (a reading `rejudge --s1` applies, the hand run of `judge-design/S1-v5.sh`) in the ruler's ledger, so the next test takes
    new draws. False when the ruler is missing or the test is already there."""
    ruler = Path(ruler)
    if not ruler.is_dir() or reading.get("test") is None:
        return False
    ledger = s1_ledger(ruler)
    if any(int(t["test"]) == int(reading["test"]) and t.get("candidate") == reading.get("candidate") for t in ledger["tests"]):
        return False
    ledger["tests"].append({"test": int(reading["test"]), "candidate": reading.get("candidate"), "candidate_dir": cand_dir_name or reading.get("candidate_dir"),
                            "champion": reading.get("champion"), "ruler_hash": reading.get("ruler_hash"), "started": None, "done": utc_now(),
                            "mean": reading.get("mean"), "se": reading.get("se"), "source": "registered"})
    write_json(ruler / S1_LEDGER_FILE, ledger)
    return True


def standing_reading(stored: Path, expect: Mapping[str, Any]) -> dict[str, Any] | None:
    """The reading stored in a candidate's folder (`s1.json`) when it is complete and names this ruler hash, champion and candidate network, else None."""
    if not Path(stored).is_file():
        return None
    try:
        old = read_json(stored)
        again = {**old, "match": keep_v2_module().s1_match(old, expect)}
        return again if again["match"]["verified"] and again.get("complete") else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


# ------------------------------------------------------------------ the base block (D68 addendum 7 item 2, lane 25)
#
# A champion that was just promoted has no judged arm outside the block it was selected on (its selection block is excluded by design), so S1 had no ruler to
# build and the first challenger after every promotion was judged by v3 alone, which a fight step cannot pass. With `--base-block auto` (the default under
# `--route-s auto`) `s1_stage` plays the arm the champion lacks: a **base block**, 2,000 runs of the champion alone on fresh seeds, into its own `arms/` folder
# in the shape every judged arm has, so `act3_starts.champion_arms` finds it, the ruler is built from it and S1 goes on. `--base-block off` keeps the old way
# (S1 unavailable, v3 decides).

BASE_BLOCK_MODES = ("auto", "off")
BASE_BLOCK_RUNS = 2000  # runs of a base block: the size of a judged block
BASE_BLOCK_SUFFIX = "-base"  # a base block's name ends so: `E22-2026-11-25-base`, its arm folder `E22-2026-11-25-base-a3-v1`
BASE_BLOCK_FILE = "base-block.json"  # in the arm folder: why and when the block was played (the first judgement's, kept across the job's retries)
BASE_BLOCK_RULE = "base-block-v1"


def base_block_of(value: str | None) -> str:
    """`--base-block`: "auto" (the default) or "off". Any other value is an error, not a silent off: a rule that decides a promotion is never guessed."""
    v = "auto" if value is None or str(value).strip() == "" else str(value).strip().lower()
    if v not in BASE_BLOCK_MODES:
        raise ValueError(f"unknown --base-block {value!r} (the modes are {', '.join(BASE_BLOCK_MODES)})")
    return v


@dataclass(frozen=True)
class BaseBlock:
    """What `s1_stage` needs to play a base block: its `name` (the arm folder is `<name>-<config>`), the folder its seed file goes to (the lab's `seeds`, beside
    the E-blocks) and the arms' own settings (`config`, `ascension`, `arm_kw`: the prior transport, as the judgement's whole-run arms play)."""

    name: str
    seeds_dir: Path
    config: str = "a3-v1"
    ascension: int = 10
    n: int = BASE_BLOCK_RUNS
    arm_kw: Mapping[str, Any] = field(default_factory=dict)
    lab_runs: tuple[Path, ...] = ()
    extra_dirs: tuple[Path, ...] = ()


def base_block_name(generation: int, judged: str | None = None, today: str | None = None) -> str:
    """The base block's name for the judgement of the E-block `judged`: its name without a `-r<n>` retry or extension suffix, then `-base` (`E22-2026-11-25-r1`
    -> `E22-2026-11-25-base`). The name is the same on every try of a judgement, so a job that was stopped resumes the block it had started. With no `judged`
    block: `E<generation>-<date>-base`. It is deliberately a name `resolve_e_block` never takes for a judged block (it skips every `-base` file)."""
    stem = re.sub(r"-r\d+$", "", judged) if judged else f"E{generation}-{today or datetime.now(UTC).strftime('%Y-%m-%d')}"
    return stem + BASE_BLOCK_SUFFIX


def base_block_arm(base_dir: Path | str, plan: BaseBlock) -> Path:
    """The champion's arm folder of the base block: `<champion>/arms/<name>-<config>`."""
    return Path(base_dir) / "arms" / f"{plan.name}-{plan.config}"


def base_block_seeds(plan: BaseBlock) -> list[str]:
    """The base block's seeds: derived from its name as every E-block's are (`gen0.e_block`), checked against every other known block and the lab's self-play
    runs, and written to `<seeds_dir>/<name>.json` so that later blocks are checked against them. A file that is already there must hold the same seeds."""
    import gen0  # noqa: PLC0415

    seeds = gen0.e_block(plan.name, plan.n)
    check_disjoint(seeds, known_seed_sets(plan.seeds_dir, list(plan.lab_runs) or None, plan.name, plan.extra_dirs))
    path = Path(plan.seeds_dir) / f"{plan.name}.json"
    if path.is_file():
        if gen0.load_seeds(path) != seeds:
            raise ValueError(f"{path.name} exists and does not hold the seeds its name gives")
    else:
        gen0.write_seed_file(plan.name, plan.n, path)
    return seeds


def base_block_unfinished(base_dir: Path | str, plan: BaseBlock) -> bool:
    """Whether a base block was started and not finished: its arm folder holds a `runs.jsonl` but no `summary.json`, or fewer rows than the block has seeds
    (a job that was killed, or a block that ended short). It is played on first, before the champion's arms are read, so a ruler is never built from half a block."""
    import act3_starts as a3  # noqa: PLC0415

    arm = base_block_arm(base_dir, plan)
    if not (arm / "runs.jsonl").is_file():
        return False
    return not ((arm / "summary.json").is_file() and a3.count_rows(arm) >= plan.n)


def play_base_block(base_dir: Path | str, base_agent: Agent, plan: BaseBlock, *, procs: int, judged: str | None = None, candidate: str = "",
                    play: Callable[..., list[dict]] | None = None, say: Callable[[str], None] | None = None) -> dict[str, Any]:
    """Play the champion alone on the base block: `run_arm` (the whole-run stage's own block player; `play` replaces it in tests) into
    `<champion>/arms/<name>-<config>`, resumable (a stopped job continues where it was), `procs` processes. When every seed has a row `base-block.json` is
    written in the arm folder (once: a later try of the same judgement keeps the first time) and its record returned: {rule, written, block, for_block,
    candidate, runs, config, seeds_file}. A block that ends short raises `S1Blocked` (its rows are kept and the next try finishes it)."""
    say = say or (lambda m: print(m, flush=True))
    play = play or run_arm
    seeds = base_block_seeds(plan)
    arm = base_block_arm(base_dir, plan)
    who = f"{Path(base_dir).parent.name}/{Path(base_dir).name}"
    have = len({r["seed"] for r in arm_rows(arm)}) if (arm / "runs.jsonl").is_file() else 0
    say(f"S1: the champion {who} has no arm outside the blocks it was selected on to build a ruler from: playing a base block of {len(seeds)} runs "
        f"({plan.name}, the champion alone, {int(procs)} processes" + (f", {have} rows on disk" if have else "") + ")")
    rows = play(seeds, arm, base_agent, plan.config, int(procs), plan.ascension, resume=True, **dict(plan.arm_kw))
    played = {r["seed"] for r in rows} & set(seeds)
    if len(played) < len(seeds):
        raise S1Blocked(f"the base block {plan.name} ended with {len(played)} of {len(seeds)} runs; what it played is kept and the next try finishes it")
    mark = arm / BASE_BLOCK_FILE
    if not mark.is_file():
        write_json(mark, {"rule": BASE_BLOCK_RULE, "written": utc_now(), "block": plan.name, "for_block": judged, "candidate": candidate, "runs": len(seeds),
                          "config": plan.config, "seeds_file": f"{plan.name}.json"})
    say(f"S1: the base block {plan.name} is played ({len(seeds)} runs)")
    return read_json(mark)


def base_block_record(base_dir: Path | str, plan: BaseBlock, judged: str | None, ruler_root: Path | str | None = None) -> dict[str, Any] | None:
    """The record of the base block played for the judgement of `judged` (the arm folder's `base-block.json`, when it names this block as its `for_block`) with what
    the champion's ruler holds of it (`arrivals`: the runs that reached the first Act 3 boss; `starts`, `matched`: the starts kept and the replays that matched), or
    None when no base block was played for this judgement. `arrivals` is None while no ruler holds it."""
    import act3_starts as a3  # noqa: PLC0415

    try:
        rec = read_json(base_block_arm(base_dir, plan) / BASE_BLOCK_FILE)
    except (OSError, ValueError):
        return None
    if not isinstance(rec, dict) or rec.get("for_block") != judged:
        return None
    meta = usable_ruler(s1_ruler_dir(base_dir, ruler_root), Path(base_dir))
    entry = a3.ruler_arms(meta).get(plan.name) if meta else None
    return {**rec, "arrivals": entry.get("arrivals") if entry else None,
            "starts": (int(entry.get("first") or 0) + int(entry.get("second") or 0)) if entry else None, "matched": entry.get("matched") if entry else None}


def base_block_note(rec: Mapping[str, Any] | None, s1_read: bool = True) -> str | None:
    """The ledger `guard` cell's words for a base block: `S1 on a base block played <ts>: N runs, M arrivals` (an S1 reading was made on the ruler it gave), or
    `a base block was played <ts>: N runs, no S1 reading` when the stage could not go on after it. None when no base block was played for the judgement."""
    if not rec:
        return None
    if s1_read and rec.get("arrivals") is not None:
        return f"S1 on a base block played {rec['written']}: {rec['runs']} runs, {rec['arrivals']} arrivals"
    return f"a base block was played {rec['written']}: {rec['runs']} runs, no S1 reading"


def s1_stage(cand: Path | str, base_dir: Path | str, cand_agent: Agent, base_agent: Agent, *, procs: int = S1_PROCS, ruler_root: Path | str | None = None,
             python: str | None = None, run: Callable[..., Any] | None = None, say: Callable[[str], None] | None = None, block: str | None = None,
             base_block: BaseBlock | None = None, play: Callable[..., list[dict]] | None = None) -> dict[str, Any]:
    """S1 for a fight-only swap: the first stage of `promote` under `--route-s auto`. Returns the reading (`keep_v2.s1_reading`, checked against this pair),
    written to the candidate's folder as `s1.json`.

    1. The champion's ruler (`D:/spireline/referee/gym-loop/arrival/<champion folder name>/`, `rl/act3_starts.py arrivals`): used when its `meta.json` names this
       champion and its starts file hashes to the meta's hash; else built there, from the champion's arms, on `procs` processes. A ruler that lacks arms the
       champion now has (`act3_starts.ruler_gap`: a block it played since, or one that grew) is **extended** first: only those arms are replayed, every start
       the ruler holds is kept, and the hash is the whole set's. `block` is the block being judged now: the champion's arm on it is left out (it is still being
       played, and the ruler must not change between two tries of one judgement). A champion with no arm to read has no ruler: `S1Unavailable`, and no folder is
       made for it, unless `base_block` (`--base-block auto`, `BaseBlock`) is given: the champion then plays a **base block** first (`play_base_block`: 2,000
       runs of the champion alone on fresh seeds, into `arms/<name>-<config>`, the whole-run stage's own block player, resumable) and the ruler is built from that
       arm. A base block that was started and not finished is finished first, whatever else the champion has.
    2. The test index k: the candidate's own when the ruler's ledger already holds a test for this fight network (a resumed or repeated judgement plays
       the same draws, and one whose `s1.json` stands for this ruler, champion and candidate is not played again); else one more than any test played
       against the ruler (`s1_used_tests`), so no draw (6k to 6k + 5) is used by two candidates. The ledger is written before the games are.
    3. `rl/act3_starts.py arrival-exam` of the candidate against the champion on the ruler, test k, `procs` processes, the report in `<candidate>/s1/`.

    Both heavy steps run as children of this process, so they run at its priority (the train job's: BelowNormal), hold no QUIET.lock line of their own and
    are told to ignore the lab's own `lab-train` line (`--allow-quiet`): the exam's `blocked_reason` would read it as another's and stop. They touch nothing
    of the gym daemon. A stop or a block raises `S1Blocked` (the job fails, the runner retries it, the work resumes); a pair S1 cannot be played for raises
    `S1Unavailable`. `run` (default `subprocess.run`), `python` (default this interpreter) and `play` (default `run_arm`, the base block's player) are replaced by
    tests."""
    say = say or (lambda m: print(m, flush=True))
    run = run or subprocess.run
    py = python or sys.executable
    cand, base_dir = Path(cand), Path(base_dir)
    ruler = s1_ruler_dir(base_dir, ruler_root)
    script = str(ROOT / "rl" / "act3_starts.py")
    lib = os.environ.get("STS2_LIB_PATH")
    allow = ["--allow-quiet", LAB_TRAIN_TAG, *(["--lib", lib] if lib else [])]
    import act3_starts as a3  # noqa: PLC0415

    same = s1_expect(cand_agent, base_agent, None)
    if same["candidate"] == same["champion"]:  # before any ruler is built or extended: nothing to compare
        raise S1Unavailable("the candidate and the champion hold the same fight network: nothing to compare")
    stored = cand / S1_FILE
    meta = usable_ruler(ruler, base_dir)
    if meta is not None:
        sync_ruler_exclusions(ruler, say)  # a ruler built by hand reaches the practice the first time a stage reads it
        # this candidate was read against this ruler already: the reading stands, and the ruler is not grown under it (a second block of the same
        # judgement, v3's extension, reads the same reading rather than a new test on a bigger ruler)
        standing = standing_reading(stored, s1_expect(cand_agent, base_agent, meta["ruler_hash"]))
        if standing is not None:
            say(f"S1: {S1_FILE} stands for this ruler, champion and candidate: not played again")
            return standing
    gap = a3.ruler_gap(meta, base_dir, [block] if block else ())  # the champion's arms the ruler lacks (all of them when there is no ruler)
    if base_block is not None and (base_block_unfinished(base_dir, base_block) or (meta is None and not gap)):  # no arm to read: the champion plays one, alone
        play_base_block(base_dir, base_agent, base_block, procs=procs, judged=block, candidate=cand.name, play=play, say=say)
        gap = a3.ruler_gap(meta, base_dir, [block] if block else ())
    if meta is None and not gap:
        raise S1Unavailable(f"the champion {base_dir.parent.name}/{base_dir.name} has no arm outside the blocks it was selected on to build a ruler from")
    if gap:
        who = f"{base_dir.parent.name}/{base_dir.name}"
        say(f"S1: building the ruler of {who} in {ruler.name} ({len(gap)} arms, {procs} processes)" if meta is None else
            f"S1: extending the ruler of {who} in {ruler.name} ({meta['starts']} starts) by {', '.join(g['label'] for g in gap)} ({procs} processes)")
        rc = run([py, script, "arrivals", "--champion", str(base_dir), "--out", str(ruler), "--procs", str(int(procs)),
                  *(["--exclude-block", block] if block else []), *allow], check=False).returncode
        if rc == 3:
            raise S1Blocked("the ruler's replays were stopped or blocked; what they wrote is kept")
        if rc == 2:
            raise S1Unavailable("the ruler cannot be " + ("built" if meta is None else "extended") + ": no arm to read, or an arm another agent played")
        if rc != 0:
            raise RuntimeError(f"the ruler's builder ended with status {rc}")
        meta = usable_ruler(ruler, base_dir)
        if meta is None:
            raise RuntimeError(f"the ruler's builder left no usable ruler in {ruler.name}")
        sync_ruler_exclusions(ruler, say)  # before the exam plays a start of it: the practice must not draw these fights
    expect = s1_expect(cand_agent, base_agent, meta["ruler_hash"])
    standing = standing_reading(stored, expect)  # the ruler was just extended, or the first look found no usable one
    if standing is not None:
        say(f"S1: {S1_FILE} stands for this ruler, champion and candidate: not played again")
        return standing
    ledger = s1_ledger(ruler)
    entry = next((t for t in ledger["tests"] if t.get("candidate") == expect["candidate"] and t.get("champion") == expect["champion"]
                  and t.get("ruler_hash") == expect["ruler_hash"] and t.get("done") is None), None)
    if entry is None:
        used = s1_used_tests(ruler, ledger)
        entry = {"test": max(used) + 1 if used else 0, "candidate": expect["candidate"], "candidate_dir": cand.name, "champion": expect["champion"],
                 "ruler_hash": expect["ruler_hash"], "started": utc_now(), "done": None}
        ledger["tests"].append(entry)
        write_json(ruler / S1_LEDGER_FILE, ledger)
    k = int(entry["test"])
    report_dir = cand / S1_EXAM_DIR
    say(f"S1: the arrival exam, test {k} (draws {6 * k} to {6 * k + 5}), {cand.name} against {base_dir.name} on {meta['starts']} starts ({procs} processes)")
    rc = run([py, script, "arrival-exam", "--ruler", str(ruler), "--candidate", str(cand), "--champion", str(base_dir), "--test", str(k), "--procs", str(int(procs)),
              "--out", str(ruler / "exam"), "--report-dir", str(report_dir), *allow], check=False).returncode
    if rc == 2:
        raise S1Unavailable("the candidate and the champion hold the same fight network: nothing to compare")
    if rc == 3:
        raise S1Blocked("the arrival exam was stopped, blocked or left incomplete; its games are kept")
    if rc != 0:
        raise RuntimeError(f"the arrival exam ended with status {rc}")
    reading = keep_v2_module().s1_reading(read_json(report_dir / "results.json"), expect)
    write_json(stored, reading)
    entry.update(done=utc_now(), mean=reading.get("mean"), se=reading.get("se"))
    write_json(ruler / S1_LEDGER_FILE, ledger)
    say(f"S1: {reading['mean']:+.2f} passes per 100 arrivals (SE {reading['se']:.2f}, {reading['starts']} starts)" if reading.get("mean") is not None
        else "S1: the exam gave no reading")
    return reading


def s1_from_file(path: Path | str, cand: Path | str, base_dir: Path | str, cand_agent: Agent, base_agent: Agent, *, ruler_root: Path | str | None = None,
                 register: bool = True) -> dict[str, Any]:
    """A reading from an arrival exam's `results.json` (`rejudge --s1`: the file `judge-design/S1-v5.sh` writes), checked against this pair: its ruler hash against
    the champion's ruler in the ruler folder (`usable_ruler`; a ruler that cannot be read leaves the hash unchecked and the reading unapplied), its candidate
    and champion against the two fight networks. With `register` the test is entered in the ruler's ledger (the next test then takes new draws)."""
    ruler = s1_ruler_dir(base_dir, ruler_root)
    meta = usable_ruler(ruler, Path(base_dir))
    if meta and register:  # a ruler built outside the train job (the hand run of 2026-11-22) is put on the practice's list the first time a stage reads it
        sync_ruler_exclusions(ruler, strict=False)
    reading = keep_v2_module().s1_reading(read_json(path), s1_expect(cand_agent, base_agent, meta["ruler_hash"] if meta else None))
    if register and meta and reading["match"]["verified"]:
        register_s1_test(ruler, reading, Path(cand).name)
    return reading


def outcome_row(row: Mapping[str, Any]) -> dict[str, Any]:
    """The part of a run's row that says how it ended (`OUTCOME_FIELDS`): what a replay must reproduce for the same seed."""
    return {k: row.get(k) for k in OUTCOME_FIELDS}


def base_arm_reuse(seeds: Sequence[str], cand: Path | str, base_dir: Path | str, base_agent: Agent, arm: str, config: str, procs: int, ascension: int,
                   *, n: int = BASE_REUSE_SEEDS, arm_kw: Mapping[str, Any] | None = None, say: Callable[[str], None] | None = None,
                   play: Callable[..., list[dict]] | None = None) -> dict[str, Any]:
    """Before S2 reuses the champion's stored arm on a block the candidate never played: replay `n` of its seeds with the champion on the current tree and
    compare the outcome rows (`OUTCOME_FIELDS`) with the stored ones. The seeds checked are the first `n` seeds of the block, in order and without a gap,
    that the champion's arm holds and the candidate's does not (each keeps its index in the block: the agent's random numbers are [agent seed, index]);
    they are played into `<candidate>/arms-check/<arm>` with the arm's own prior transport (`arm_kw`). When every one is the same the stored arm is reused
    (`action` "reused", what `run_arm` does by itself); when one differs the stored arm is moved to `<champion>/arms-stale/` and the champion plays the
    block afresh (`action` "played afresh": an engine change since it was played means its rows are not this tree's). No row the candidate did not play
    beside it, nothing to check (`action` "none": a resumed block, or a fresh one). The record goes to promotion.json (`base_reuse`)."""
    say = say or (lambda m: print(m, flush=True))
    play = play or run_arm
    cand, base_dir = Path(cand), Path(base_dir)
    stored_dir, mine_dir = base_dir / "arms" / arm, cand / "arms" / arm
    stored = {r["seed"]: r for r in arm_rows(stored_dir)} if (stored_dir / "runs.jsonl").is_file() else {}
    mine = {r["seed"] for r in arm_rows(mine_dir)} if (mine_dir / "runs.jsonl").is_file() else set()
    reusable = [s for s in seeds if s in stored and s not in mine]
    rec: dict[str, Any] = {"arm": arm, "stored_rows": len(stored), "candidate_rows": len(mine & set(seeds)), "reusable": len(reusable), "wanted": int(n),
                           "fields": list(OUTCOME_FIELDS)}
    if not reusable:
        return {**rec, "checked": 0, "same": 0, "differing": [], "action": "none",
                "why": "the champion's stored arm holds no row the candidate did not play beside it: nothing is reused unchecked"}
    first = list(seeds).index(reusable[0])
    window: list[str] = []
    ok = set(reusable)
    for s in list(seeds)[first:]:
        if s not in ok or len(window) >= n:
            break
        window.append(s)
    say(f"S2: before the champion's stored arm on {arm} is reused, {len(window)} of its seeds are replayed on this tree")
    replay = play(window, cand / "arms-check" / arm, base_agent, config, procs, ascension, start_ix=first, **dict(arm_kw or {}))
    got = {r["seed"]: r for r in replay}
    differing = [s for s in window if s not in got or outcome_row(got[s]) != outcome_row(stored[s])]
    rec.update(checked=len(window), same=len(window) - len(differing), differing=differing, start_index=first)
    if not differing:
        return {**rec, "action": "reused", "why": f"{len(window)} of {len(window)} replayed seeds gave the stored outcome rows"}
    stale = base_dir / "arms-stale" / f"{arm}-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}"
    stale.parent.mkdir(parents=True, exist_ok=True)
    os.replace(stored_dir, stale)
    return {**rec, "action": "played afresh", "stale": f"{base_dir.name}/arms-stale/{stale.name}",
            "why": f"{len(differing)} of {len(window)} replayed seeds differ from the stored rows: the stored arm is set aside and the champion plays the block again"}


def backup_arm_rows(arm_dirs: Sequence[Path | str]) -> list[str]:
    """A copy of each arm's `runs.jsonl` beside it (`runs.before-completion-<UTC>.jsonl`; nothing reads that name) before a completion plays into the arm: the
    funnel's resume (`gen0.resume_rows`) rewrites the file with the rows whose identity still matches and starts it empty when the agent or the setting has
    changed, so the stored rows of a block are kept safe first. The names of the copies written."""
    import shutil  # noqa: PLC0415

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    made: list[str] = []
    for d in arm_dirs:
        src = Path(d) / "runs.jsonl"
        if src.is_file():
            dst = src.with_name(f"runs.before-completion-{stamp}.jsonl")
            shutil.copyfile(src, dst)
            made.append(f"{Path(d).parent.parent.name}/arms/{Path(d).name}/{dst.name}")
    return made


def sequential_doc(seq: Mapping[str, Any], seeds: Sequence[str], *, smoke: int, batch: int, played: int, fut_min: int | None, v2: bool, rule: str,
                   frontier_name: str | None, route_s: bool = False) -> dict[str, Any]:
    """The `sequential` entry of a promotion.json: the funnel's schedule, stop and looks (what `promote` writes for a block played as the funnel; `rejudge`
    writes it when it completes a block). Under route S the futility stop is off and the entry says so."""
    schedule = seq_looks(len(seeds), smoke, batch)
    return {"rule": SEQ_RULE, "block_seeds": len(seeds), "played": played, "smoke": smoke, "batch": batch, "schedule": schedule,
            "z_early": (keep_v2_module().early_z if v2 else seq_early_z)(len(schedule)), "refuse_alpha": SEQ_REFUSE_ALPHA, "early_alpha": SEQ_EARLY_ALPHA,
            "stop": seq["stop"], "looks": seq["looks"], "futility": {"enabled": fut_min is not None and not route_s, "min_seeds": fut_min},
            **({"keep_rule": rule, "frontier": frontier_name} if v2 else {}), **({"route": "S"} if route_s else {})}


def promote(cand: Path, base: Path | None, seeds_file: Path, *, agent_root: Path, procs: int = 16, config: str = "a3-v1",
            ascension: int = 10, lab: Path | None = None, lab_runs: list[Path] | None = None, boot: int = BOOT_B,
            review_guard: bool | None = None, server_opts=None, sequential: bool = False, smoke: int = SEQ_SMOKE,
            batch: int = SEQ_BATCH, keep_rule: str | None = None, futility: bool | None = None,
            futility_min_seeds: int | None = None, screen: bool | None = None, pool_blocks: Sequence[str] = (),
            keep_rule_why: str | None = None, min_gain: float | None = None, route_s: str | None = None,
            base_block: str | None = None) -> dict[str, Any]:
    """Play candidate and base on the same E-block seeds, test, and record the decision. With the review guard on
    (`review_guard`, default the SPIRELINE_REVIEW_GUARD variable; off) the decision is `decide_review`'s and the tsv row carries the attempt
    and the guard's verdict. A third attempt of a generation is played and judged like any other (D65 addendum 2): the old cap of two
    is recorded in the decision (`checks["attempt"]`, `attempt_advisory`), in the verdict's attempt block and in the row's guard cell, and
    refuses nothing.
    `server_opts` puts both arms on a prior server each (one policy per server); the default is the CPU path.

    With `sequential` (the funnel, D51; see `seq_call` for the rule and its error rates) the block is played in looks: the first
    `smoke` seeds of the E-block, then `batch` more at a time, both arms to each look, and the block stops at a look that refuses
    (or promotes early); a block that is not stopped plays every seed and is judged exactly as without `sequential`. A stop is
    recorded like any verdict: promotion.json (`n_seeds` the seeds played, `sequential` the looks and the stop), a promotions.tsv row
    (`seeds` the seeds played, the `guard` cell ending with the stop) and, on a refusal, the attempt spent as for a block refusal:
    the attempt is the number of distinct candidates (combat prior sha, run head sha) of the generation in promotions.tsv
    (`attempts_before`), counted from this row on; nothing else holds it. A killed job resumes into the same arm folders at the first look it has not finished.

    `keep_rule` "v2" (default `keep_rule_of`: the SPIRELINE_KEEP_RULE variable, else "v1") decides by the keep test v2
    (tools/lab/keep_v2.py, docs/rl/KEEP_TEST_V2.md) instead of D24 and the review guard: the frontier is read from the incumbent's own arms
    before anything is played (`incumbent_frontier`), the funnel calls each look on the floor and deep_f, the cap and the drift check
    are part of the rule (the review-guard flag is not needed), and promotion.json gains `keep_v2` and the tsv row its nine columns.

    `keep_rule_why` (2026-11-19): when the train job chose the rule between a marker's and the lab's (`jobs.choose_keep_rule`), its one-line reason
    goes to promotion.json (`keep_rule_choice`) and to the end of the row's `guard` cell, so the verdict says which rule judged it and why.

    `min_gain` (2026-11-19, D68 addendum 3; `--min-gain`) is route A's smallest deep_f gain per 100 runs under the keep rules, `keep_v2.MIN_GAIN` (+1.0) when
    None. It goes to the verdict (`keep_v2.min_gain`), to the decision (`decision.min_gain`, only when it is not +1.0, so a default decision is
    as it was) and to the ledger's `guard` cell (`v3 min gain 0.5 route A | ...`, only then), and to the funnel's looks and futility stop. Under
    v1 only the default is accepted.

    `keep_rule` "v3" (D66, docs/rl/KEEP_TEST_V3.md) is the same machinery with the keep rule v3's verdict (`keep_v2.judge_v3`): route A alone
    promotes, the floors refuse only a collapse, the drift check is advisory, and a final judgement whose deep_f gain is at least +1.0 per 100
    but whose 0.005 bound is not above 0 is an *extension*: decision `extend` (promotion.json `decision_text`, the ledger's `decision`
    column), route X, nothing promoted. `pool_blocks` (v3 only) names earlier blocks whose arms, the candidate's and the base's, are pooled with
    this block's: the judgement is route A on all the pairs (the funnel's looks and its futility's N count them too), and promotion.json's
    `pooled` names every block and its pairs. The pooled judgement is the second look: a candidate that was extended is not extended again.

    The futility stop (`futility`, default on; `futility_min_seeds`, default FUTILITY_MIN_SEEDS; only with `sequential`; docs/rl/KEEP_TEST_V2.md):
    from that many paired seeds on, a look that would play on also asks whether any route's 95% upper bound of the candidate's gain
    reaches the gain the route needs to promote on the whole block (`futility_check`); when none does the block ends there, rejected,
    with the decision's rule `futility@<seeds>`, the arms' rows kept, the keep test v2 run on the seeds played as for a refusal, and a
    line in the job log naming the bounds that fell short.

    The screens (`screen`, default on; rl/screens.py): before an arm is played the candidate's folder is compared with the incumbent's. Only the
    fight network different: the candidate plays the fight exam's human stratum and is rejected (rule `screen:fight`) when its survival is below the
    incumbent's by more than two standard errors. Only the human prior different: both are scored on the held-out human choices and the candidate is
    rejected (`screen:prior`) when its log-loss is worse. A rejection writes promotion.json (`screen` holds the figures) and a ledger row with seeds 0
    and that rule, spends the attempt, plays nothing; a screen that passes, is skipped (and says why) or fails goes on to the arms with its figures
    in promotion.json and the row's `guard` cell.

    `route_s` ("auto" or "off", the default; `--route-s`; D68 addendum 6 outcome 4, docs/rl/KEEP_TEST_V3.md "Route S"; the keep rule v3's): under "auto" a
    candidate whose agent.json says `built.swap: fight` (`fight_only_swap`: `rl/gym_wiring.py`'s champion with only its fight network replaced) is first read
    by S1, the arrival exam (`s1_stage`, after the screen and before any arm is played; its reading is `<candidate>/s1.json` and promotion.json's `s1`).
    When S1 holds, S2 is the funnel over the whole block with its looks refusing only (`funnel(route_s=True)`: no futility stop, no early promotion, no
    extension), the champion's stored arm is reused on a block the candidate never played only after 20 of its seeds replay identically (`base_arm_reuse`,
    promotion.json's `base_reuse`), and the verdict is route S's (`keep_v2.judge_v3(s1=)`). When S1 does not hold, or cannot be played for this pair,
    the judgement is v3's exactly as before, with the reason recorded (`route_s_mode`). Any other candidate is judged as before.

    `base_block` ("auto", the default, or "off"; `--base-block`; D68 addendum 7 item 2, only under `route_s` "auto"): a champion with no judged arm outside the blocks it
    was selected on (just promoted) plays a **base block** first, 2,000 runs of the champion alone on fresh seeds (`s1_stage(base_block=)`, `play_base_block`: the block
    `<E-block>-base`, its seeds file beside the E-block's, its arm in the champion's `arms/`), and S1 is read on the ruler that arm gives. The ledger's `guard` cell says
    `S1 on a base block played <ts>: N runs, M arrivals` and promotion.json's `route_s_mode.base_block` holds the record. "off" is the old way: S1 unavailable, v3 decides."""
    import gen0  # noqa: PLC0415

    rule = keep_rule_of(keep_rule)
    v2 = rule in KEEP_TESTS
    if pool_blocks and rule != "v3":
        raise ValueError("pooling blocks is the keep rule v3's (--keep-rule v3)")
    mode = route_s_of(route_s)
    base_mode = base_block_of(base_block)
    if mode == "auto" and rule != "v3":
        raise ValueError("route S is the keep rule v3's (--keep-rule v3)")
    gain = min_gain_of(min_gain, rule)  # None under v1; the keep rules' route A minimum otherwise
    fut_min = None if futility is False or (futility is None and not FUTILITY_ON) else int(futility_min_seeds or FUTILITY_MIN_SEEDS)
    seeds = gen0.load_seeds(seeds_file)
    name = Path(seeds_file).stem
    if name.startswith("SP-"):
        raise ValueError("an E-block is never named like a self-play stream")
    check_disjoint(seeds, known_seed_sets(Path(seeds_file).parent, lab_runs, name, [lab_seed_dir(lab)] if lab else ()))
    cand_agent = load_agent(cand)
    base_agent = load_agent(base, agent_root=None if base else agent_root)
    cand_sha, head_sha = candidate_identity(cand_agent)
    recipe = candidate_recipe(cand_agent)
    tsv = Path(agent_root) / "promotions.tsv"
    if block_used_by_another(tsv, name, cand_sha, head_sha, recipe):  # a second look at one block is a look the alpha does not pay for
        raise ValueError(f"E-block {name} already judged another candidate; make a fresh one (resolve_e_block)")
    rows = read_promotions(tsv)
    guard_on = review_guard_on(review_guard)
    attempt = attempts_before(rows, cand_agent.generation, cand_sha, head_sha, recipe) + 1
    if (guard_on or v2) and attempt > review_guard_module().ATTEMPT_CAP:  # D65 addendum 2: the old cap is recorded, not gating
        print(f"generation {cand_agent.generation}: attempt {attempt}, over the old cap of {review_guard_module().ATTEMPT_CAP}; "
              "the cap is recorded and gates nothing, so the block is played", flush=True)
    frontier_info = incumbent_frontier(agent_root, rows, config, block=[name, *pool_blocks]) if v2 else None  # fixed before a seed is played
    guard_on = guard_on and not v2  # the keep test v2 carries its own guards
    arm = f"{name}-{config}"
    arm_kw = {"server_opts": server_opts} if server_opts is not None else {}  # the CPU path passes nothing new
    base_dir = Path(base_agent.path) if base_agent.path else Path(agent_root) / "gen0"
    pool_c, pool_b, pool_info = pooled_arm_rows(Path(cand), base_dir, config, list(pool_blocks), taken=seeds) if pool_blocks else ([], [], [])  # before a seed is played
    pre_n = len({r["seed"] for r in pool_c} & {r["seed"] for r in pool_b})  # the pooled pairs
    may_extend = rule == "v3" and not pool_blocks and not extended_before(rows, cand_sha, head_sha, recipe, name)  # an extension is still available
    screen_doc = run_screen(Path(cand), base_dir, cand_agent, base_agent, procs=procs) if (SCREEN_ON if screen is None else screen) else None
    if screen_doc is not None and screen_doc.get("rejected") and not screen_doc.get("skipped"):
        return screen_rejection(Path(cand), cand_agent, base_agent, screen_doc, name=name, config=config, procs=procs, agent_root=Path(agent_root),
                                attempt=attempt, cand_sha=cand_sha, head_sha=head_sha, recipe=recipe, rule=rule if v2 else None, lab=lab)
    s1: dict[str, Any] | None = None  # route S: S1's reading for this pair (None: not asked, or not playable)
    route_s_mode: dict[str, Any] | None = None
    base_rec: dict[str, Any] | None = None  # the base block played for this judgement, with what the ruler holds of it
    if mode == "auto":
        route_s_mode = {"mode": "auto", "fight_swap": fight_only_swap(cand)}
        if not route_s_mode["fight_swap"]:
            route_s_mode["why"] = "not a fight-only swap (agent.json built.swap does not read 'fight'): v3 decides"
            route_s_mode["reason"] = "not a fight-only swap"
        else:
            base_plan = (BaseBlock(name=base_block_name(cand_agent.generation, name), seeds_dir=lab_seed_dir(lab) if lab else Path(seeds_file).parent, config=config,
                                   ascension=ascension, n=BASE_BLOCK_RUNS, arm_kw=arm_kw, lab_runs=tuple(Path(x) for x in lab_runs or ()),
                                   extra_dirs=(lab_seed_dir(lab),) if lab else ()) if base_mode == "auto" else None)
            try:
                s1 = s1_stage(cand, base_dir, cand_agent, base_agent, procs=min(int(procs), S1_PROCS), say=lambda m: print(m, flush=True), block=name,
                              **({"base_block": base_plan} if base_plan is not None else {}))
            except S1Unavailable as why:
                route_s_mode["why"] = f"S1 could not be played ({why}): v3 decides"
                route_s_mode["reason"] = str(why)
            base_rec = base_block_record(base_dir, base_plan, name) if base_plan is not None else None
            if base_rec:
                route_s_mode["base_block"] = base_rec
    s_on = bool(s1 is not None and keep_v2_module().route_s1(s1, fight_swap=True)["holds"])  # S1 holds: S decides and S2 is the funnel
    if route_s_mode is not None:
        route_s_mode["applies"] = s_on
    reuse = (base_arm_reuse(seeds, Path(cand), base_dir, base_agent, arm, config, procs, ascension, arm_kw=arm_kw,
                            say=lambda m: print(m, flush=True)) if s_on else None)  # before an arm is played: the stored arm is reused only if it replays
    seq: dict[str, Any] | None = None

    def keep_of(cr: list[dict], br: list[dict], final: bool = False) -> dict[str, Any] | None:
        return (keep_v2_verdict(cr, br, rows, cand_gen=cand_agent.generation, cand_sha=cand_sha, block=name, frontier_info=frontier_info,
                                boot=boot, head_sha=head_sha, recipe=recipe, rule=rule, pooled=list(pool_blocks), final=final,
                                min_gain=gain, **({"s1": s1} if s1 is not None else {})) if v2 else None)

    if sequential:
        def verdict(cr: list[dict], br: list[dict]) -> dict[str, Any] | None:
            return (review_guard_verdict(cr, br, rows, cand_gen=cand_agent.generation, cand_sha=cand_sha, block=name, boot=boot,
                                         head_sha=head_sha, recipe=recipe) if guard_on else None)

        def rows_at(n: int) -> tuple[list[dict], list[dict]]:
            part = seeds[:n]
            return (run_arm(part, Path(cand) / "arms" / arm, cand_agent, config, procs, ascension, resume=True, **arm_kw),
                    run_arm(part, base_dir / "arms" / arm, base_agent, config, procs, ascension, resume=True, **arm_kw))

        seq = funnel(seeds, rows_at, lambda cr, br: compare_arms(cr, br, ALPHA, boot, verdict(cr, br), **({"keep": keep_of(cr, br)} if v2 else {})),
                     smoke=smoke, batch=batch, futility=fut_min, say=lambda m: print(m, flush=True),
                     **({"keep_rule": rule, "frontier": frontier_info["index"], "frontier_name": frontier_info["name"],
                         "prefix": (pool_c, pool_b), "extend_possible": may_extend and not s_on, "min_gain": gain,
                         **({"route_s": True} if s_on else {})} if v2 else {}))
        cand_rows, base_rows, played = seq["cand_rows"], seq["base_rows"], seq["stop"]["n"]  # (the funnel's rows hold the pooled blocks' too)
    else:
        cand_rows = run_arm(seeds, Path(cand) / "arms" / arm, cand_agent, config, procs, ascension, **arm_kw)
        base_rows = run_arm(seeds, base_dir / "arms" / arm, base_agent, config, procs, ascension, **arm_kw)
        played = len(seeds)
        cand_rows, base_rows = cand_rows + pool_c, base_rows + pool_b
    stopped_early = seq is not None and seq["stop"]["action"] != "final"
    guard = (review_guard_verdict(cand_rows, base_rows, rows, cand_gen=cand_agent.generation, cand_sha=cand_sha, block=name, boot=boot,
                                  head_sha=head_sha, recipe=recipe) if guard_on else None)
    keep = keep_of(cand_rows, base_rows, final=not stopped_early)  # only a judgement of every seed of the block extends
    res = compare_arms(cand_rows, base_rows, ALPHA, boot, guard, **({"keep": keep} if v2 else {}))  # a call under v1 is as it always was
    if res["paired_seeds"] != played + pre_n:  # a partial arm must not promote on fewer pairs than the block
        raise ValueError(f"only {res['paired_seeds']} of {played + pre_n} seeds are paired in both arms")
    if stopped_early:  # the funnel's own check joins the others, so the decision names why the block ended before its last seed
        d = res["decision"]
        d["checks"]["sequential"] = seq["stop"]["action"] not in STOP_FAILS
        d["promote"] = checks_hold(d["checks"], d.get("advisory_checks"))
        d["rule"] = f"{d['rule']}+{SEQ_RULE}" + (f"+{seq['stop']['rule']}" if seq["stop"]["action"] == "futility" else "")
    doc = {"written": utc_now(), "seed_block": name, "n_seeds": played + pre_n, "config": config, "procs": procs,
           "alpha": ALPHA, "cand": {"agent": cand_agent.describe(), "summary": arm_summary(cand_rows)},
           "base": {"agent": base_agent.describe(), "summary": arm_summary(base_rows)}, **res,
           "decision_text": decision_text(res["decision"]), "cand_sha": cand_sha,
           "run_head_sha": head_sha, **({"macro": recipe} if recipe else {}), **({"keep_rule": rule} if v2 else {}),
           **({"keep_rule_choice": {"rule": rule, "why": keep_rule_why}} if keep_rule_why else {}),
           # DESIGN 5.3/5.4 ask for both; PLAN section 5 does not, so a promotion here is not the full DESIGN gate
           "f7": "not run", "matched_wall_clock": False}
    if pool_blocks:
        own_seeds = set(seeds)
        doc["pooled"] = pooled_record(name, played, ([r for r in cand_rows if r["seed"] in own_seeds], [r for r in base_rows if r["seed"] in own_seeds]), pool_info)
    if seq is not None:
        doc["sequential"] = sequential_doc(seq, seeds, smoke=smoke, batch=batch, played=played, fut_min=fut_min, v2=v2, rule=rule,
                                           frontier_name=frontier_info["name"] if v2 else None, route_s=s_on)
        if stopped_early:  # the figures above are those of the seeds played, not of the whole block
            doc["sequential"]["note"] = (f"stopped at {played} of {len(seeds)} seeds: floor, boss_kill and wins are the figures at the stop; "
                                         "the mean of a block that stopped on a bound is biased away from 0 (the decision's error rates hold)")
    if v2:
        doc["human_bar"] = human_bar_note(cand_agent, base_agent)  # report-only: no rule reads it
    if screen_doc is not None:
        doc["screen"] = screen_doc
    if route_s_mode is not None:  # route S: what was asked, S1's reading, how far the whole run moved from it, and the check on the champion's stored arm
        doc["route_s_mode"] = route_s_mode
        if s1 is not None:
            doc["s1"] = s1
        shift = ((keep or {}).get("route_s") or {}).get("state_shift")
        if shift:
            doc["state_shift"] = shift
        if reuse is not None:
            doc["base_reuse"] = reuse
    write_json(Path(cand) / "promotion.json", doc)
    f, bk = doc["floor"], doc["boss_kill"]
    cells = keep_cells(keep, attempt) if v2 else guard_cells(guard, attempt)
    if pool_blocks:
        cells["guard"] = f"{cells['guard']} | {pooled_say(doc['pooled'])}".replace("\t", " ")
    if screen_doc is not None and screen_doc.get("line"):  # a screen that passed leaves its figures in the row
        cells["guard"] = f"{cells['guard']} | {screen_doc['line']}".replace("\t", " ")
    if keep_rule_why:  # the train job chose the rule between the marker's and the lab's: the row says which and why
        cells["guard"] = f"{cells['guard']} | keep rule {rule}: {keep_rule_why}".replace("\t", " ")
    if route_s_note(route_s_mode, s1):  # route S was asked and had no reading: the row says so (a reading prints its own parts)
        cells["guard"] = f"{cells['guard']} | {route_s_note(route_s_mode, s1)}".replace("\t", " ")
    if base_rec:  # the champion played a base block for this judgement: S1 was read on the ruler it gave
        cells["guard"] = f"{cells['guard']} | {base_block_note(base_rec, s1 is not None)}".replace("\t", " ")
    if stopped_early:  # the row's guard cell carries the stop; the columns are the block's as ever
        said = (f"funnel {seq['stop']['rule']} of {len(seeds)} seeds" if seq["stop"]["action"] == "futility"
                else f"funnel {seq['stop']['action']} at {played} of {len(seeds)} seeds")
        cells["guard"] = f"{cells['guard']} | {said}".replace("\t", " ")
    append_promotion(tsv, {"written": doc["written"], "cand_gen": cand_agent.generation, "base_gen": base_agent.generation,
                           "seed_block": name, "seeds": doc["n_seeds"], "d_floor": f"{f['mean']:.4f}", "floor_lower": f"{f['lower']:.4f}",
                           "p_floor": f"{f['p']:.5f}", "boss_cand": bk["cand"], "boss_base": bk["base"], "p_boss": f"{bk['p']:.5f}",
                           "wins_cand": doc["cand"]["summary"]["wins"], "wins_base": doc["base"]["summary"]["wins"],
                           "decision": doc["decision_text"], "cand_sha": cand_sha, "cand_dir": Path(cand).name,
                           "run_head_sha": head_sha, "macro": recipe, **cells})
    if res["decision"]["promote"]:
        write_json(Path(agent_root) / "current.json", {"generation": cand_agent.generation, "dir": str(Path(cand).resolve()),
                                                        "promoted": doc["written"], "seed_block": name})
        if lab is not None:
            p = Path(lab) / "score" / "pending"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(utc_now() + "\n", encoding="utf-8")
    return doc


def decision_text(decision: Mapping[str, Any]) -> str:
    """The word a promotion.json and the ledger's `decision` column carry: "promoted", "extend" (the keep rule v3 asks for a second block; nothing
    is promoted) or "rejected"."""
    return "promoted" if decision["promote"] else "extend" if decision.get("extend") else "rejected"


def pooled_say(pooled: Mapping[str, Any]) -> str:
    """A pooled judgement's words for the ledger's `guard` cell: the pairs of each block and in all."""
    return "pooled " + " + ".join(f"{b['block']} ({b['pairs']})" for b in pooled["blocks"]) + f" = {pooled['pairs']} pairs"


def funnel_stop_record(doc: dict[str, Any] | None) -> dict[str, Any] | None:
    """The funnel's early stop recorded in a promotion.json: `{"action", "played", "block_seeds", "sequential"}` (`sequential` the
    whole `sequential` key, to be carried forward), or None for a block that was not played as the funnel, or that played every seed
    (stop action "final": judged exactly as without the funnel). A document that does not say how far it played is read as none."""
    seq = (doc or {}).get("sequential")
    if not isinstance(seq, dict) or not isinstance(seq.get("stop"), dict):
        return None
    action, played, total = seq["stop"].get("action"), seq.get("played"), seq.get("block_seeds")
    if action in (None, "final") or not isinstance(played, int) or not isinstance(total, int) or not 0 < played < total:
        return None
    return {"action": action, "played": played, "block_seeds": total, "sequential": seq}


def rejudge(cand: Path, base: Path | None, seeds_file: Path, *, agent_root: Path, config: str = "a3-v1", note: str,
            lab: Path | None = None, boot: int = BOOT_B, review_guard: bool | None = None, keep_rule: str | None = None,
            out: Path | None = None, pool_blocks: Sequence[str] = (), min_gain: float | None = None, route_s: str | None = None,
            s1_file: Path | str | None = None, complete: bool = False, procs: int = 16, ascension: int = 10, server_opts=None,
            ruler_root: Path | str | None = None) -> dict[str, Any]:
    """Apply the current rule to a candidate's finished arms on its own block, without playing again: for an owner
    decision that changes the rule after a block was judged (D24). Both cached arms must cover the whole block, except a block
    the funnel stopped early (the earlier promotion.json carries `sequential` with a stop before the last look,
    `funnel_stop_record`): its arms hold only the seeds played, the first N of the block in file order, so those are the seeds
    judged, and both arms must hold all N. The document says so (`n_seeds` N, `rejudged.seeds` the N, the block's length and the
    stop, `sequential` carried over with a note), the decision carries the funnel's own check as `promote` writes it (a refusal
    stays refused: the block ended on it) and the tsv `guard` cell ends with the stop. The earlier promotion.json is kept
    beside the new one; the tsv gets a second row for the same candidate and block. `review_guard` as in `promote` (a re-read is
    not a new attempt: the attempt number is reported against the old cap as within it; the cap gates nothing anyway, the other checks decide).

    `keep_rule` as in `promote`: under "v2" the frontier is read from the incumbent's arms on the ledger's rows before this block, and
    the keep test v2 decides (the funnel stop, if any, is carried as above). With `out`, a dry run: the judgement reads the agent folder
    (the ledger, the candidate's and the incumbent's arms, the earlier promotion.json) and writes nothing there; promotion.json, a
    copy of promotions.tsv with the new row and, on a promotion, current.json go to the `out` folder, and no lab marker is written
    (`lab` is refused).

    `keep_rule` "v3" is the keep rule v3 (D66, docs/rl/KEEP_TEST_V3.md). `pool_blocks` (v3 only; repeatable on the command line as `--pool-block`)
    names other blocks of the same two players whose finished arms (`arms/<block>-<config>` of the candidate and of the base) are pooled with
    the seed file's block: route A is judged on every pair of all the blocks, the frontier and the lineage leave all of them out, the pooled
    judgement is the second look (the candidate counts as extended, so it is promoted or refused, never extended again), and promotion.json's
    `pooled` names each block and its pairs (and each side's kills per boss). A block the funnel stopped has its verdict and is not pooled.
    The ledger row is the seed file's block with the pooled pairs in `seeds` and the pooled blocks named in the `guard` cell.

    `min_gain` as in `promote` (`rejudge --min-gain 0.5`): the minimum the judgement asks of route A, recorded in the verdict, the decision (when it is
    not +1.0) and the row's `guard` cell.

    `route_s` "auto" (`--route-s auto`; the keep rule v3's; D68 addendum 6 outcome 4) rejudges a fight-only swap under route S without playing S1 again: the
    reading is the arrival exam's `results.json` named by `s1_file` (`--s1`: checked against the champion's ruler hash, the champion's and the candidate's
    fight networks; a mismatch is not applied and the reason is recorded), else the candidate's own `s1.json`, else none (v3 decides). When S1 holds the
    decision is S's. A block the funnel stopped before its last look has too few pairs for S2's final bound: with `complete` (`--complete-block`) the
    block is completed to every seed of the file, the stored rows reused (`run_arm(resume=True)`, the funnel's own resume: a seed with a row is not played
    again) and the funnel's looks walked again under S2's rules, `procs`, `ascension` and `server_opts` as `promote` takes them; without it the rejudgement
    stops with an error that says what is missing. A completion plays games, so it is never a dry run (`out`). `ruler_root` replaces the ruler folder's parent
    (tests)."""
    import gen0  # noqa: PLC0415

    rule = keep_rule_of(keep_rule)
    v2 = rule in KEEP_TESTS
    if pool_blocks and rule != "v3":
        raise ValueError("pooling blocks is the keep rule v3's (--keep-rule v3)")
    mode = route_s_of(route_s)
    if (mode == "auto" or s1_file or complete) and rule != "v3":
        raise ValueError("route S is the keep rule v3's (--keep-rule v3)")
    if (s1_file or complete) and mode != "auto":
        raise ValueError("--s1 and --complete-block are route S's: name --route-s auto")
    gain = min_gain_of(min_gain, rule)
    if out is not None and lab is not None:
        raise ValueError("a dry run (out) writes no lab marker: do not name the lab folder")
    seeds = gen0.load_seeds(seeds_file)
    name = Path(seeds_file).stem
    cand_agent = load_agent(cand)
    base_agent = load_agent(base, agent_root=None if base else agent_root)
    arm = f"{name}-{config}"
    base_dir = Path(base_agent.path) if base_agent.path else Path(agent_root) / "gen0"
    old = Path(cand) / "promotion.json"
    before = json.loads(old.read_text(encoding="utf-8")) if old.is_file() else None
    stop = funnel_stop_record(before)
    if stop is not None and (before.get("seed_block") != name or stop["block_seeds"] != len(seeds)):
        raise ValueError(f"the funnel stopped block {before.get('seed_block')} at {stop['played']} of {stop['block_seeds']} seeds, "
                         f"but the seed file is {name} with {len(seeds)}")
    s1 = None
    route_s_mode = None
    if mode == "auto":
        route_s_mode = {"mode": "auto", "fight_swap": fight_only_swap(cand)}
        if s1_file:
            s1 = s1_from_file(s1_file, cand, base_dir, cand_agent, base_agent, ruler_root=ruler_root, register=out is None)
            route_s_mode["source"] = f"--s1 {Path(s1_file).name}"
        elif (Path(cand) / S1_FILE).is_file():
            meta = usable_ruler(s1_ruler_dir(base_dir, ruler_root), base_dir)
            if meta and out is None:
                sync_ruler_exclusions(s1_ruler_dir(base_dir, ruler_root), strict=False)
            s1 = {**read_json(Path(cand) / S1_FILE)}
            s1["match"] = keep_v2_module().s1_match(s1, s1_expect(cand_agent, base_agent, meta["ruler_hash"] if meta else None))
            route_s_mode["source"] = f"the candidate's {S1_FILE}"
        else:
            route_s_mode["why"] = f"no S1 reading (no --s1 file, no {S1_FILE} in the candidate's folder): v3 decides"
            route_s_mode["reason"] = f"no S1 reading (no --s1 file, no {S1_FILE} in the candidate's folder)"
    s_on = bool(s1 is not None and keep_v2_module().route_s1(s1, fight_swap=bool(route_s_mode and route_s_mode["fight_swap"]))["holds"])
    if route_s_mode is not None:
        route_s_mode["applies"] = s_on
    played = stop["played"] if stop is not None else len(seeds)
    earlier_stop = stop["action"] if stop is not None else None
    earlier_played = played
    if pool_blocks and stop is not None:
        raise ValueError(f"the funnel stopped block {name}: it has its verdict and is not pooled")
    if s_on and stop is not None:  # S2's final bound needs every pair: the block is completed, or the rejudgement says what is missing
        if not complete:
            raise ValueError(f"S1 holds, but the funnel stopped block {name} at {stop['played']} of {len(seeds)} seeds and S2's final bound is asked on all of "
                             f"them: add --complete-block to play the other {len(seeds) - stop['played']} pairs (the stored rows are kept)")
        if out is not None:
            raise ValueError("completing a block plays games into the agent folders: it is never a dry run (do not name --out)")
    cand_rows, base_rows = arm_rows(Path(cand) / "arms" / arm), arm_rows(base_dir / "arms" / arm)
    if stop is not None:  # the arms of a stopped block hold the seeds played and no others; judge those, by place in the block
        part = set(seeds[:played])
        cand_rows, base_rows = [r for r in cand_rows if r["seed"] in part], [r for r in base_rows if r["seed"] in part]
    pool_c, pool_b, pool_info = pooled_arm_rows(Path(cand), base_dir, config, list(pool_blocks), taken=[r["seed"] for r in cand_rows]) if pool_blocks else ([], [], [])
    pre_n = len({r["seed"] for r in pool_c} & {r["seed"] for r in pool_b})  # the pooled pairs
    own = (cand_rows, base_rows)  # the judged block's own rows, before the pooled blocks are added
    cand_rows, base_rows = cand_rows + pool_c, base_rows + pool_b
    cand_sha, head_sha = candidate_identity(cand_agent)
    recipe = candidate_recipe(cand_agent)
    ledger = Path(agent_root) / "promotions.tsv"
    rows = read_promotions(ledger)
    attempt = attempts_before(rows, cand_agent.generation, cand_sha, head_sha, recipe) + 1
    guard = (review_guard_verdict(cand_rows, base_rows, rows, cand_gen=cand_agent.generation, cand_sha=cand_sha, block=name, boot=boot,
                                  apply_cap=False, head_sha=head_sha, recipe=recipe) if review_guard_on(review_guard) and not v2 else None)
    keep = None
    walked: dict[str, Any] | None = None
    reuse = None
    backups: list[str] = []
    s1_kw = {"s1": s1, "fight_swap": bool(route_s_mode and route_s_mode["fight_swap"])} if s1 is not None else {}
    if v2:
        frontier_info = incumbent_frontier(agent_root, rows, config, block=[name, *pool_blocks])
        if s_on and stop is not None:  # complete the block: the funnel's own resume plays the seeds that have no row, S2's looks refuse only
            arm_kw = {"server_opts": server_opts} if server_opts is not None else {}
            backups = backup_arm_rows([Path(cand) / "arms" / arm, base_dir / "arms" / arm])  # before anything plays into the stored arms
            reuse = base_arm_reuse(seeds, Path(cand), base_dir, base_agent, arm, config, procs, ascension, arm_kw=arm_kw, say=lambda m: print(m, flush=True))

            def rows_at(n: int) -> tuple[list[dict], list[dict]]:
                part = seeds[:n]
                return (run_arm(part, Path(cand) / "arms" / arm, cand_agent, config, procs, ascension, resume=True, **arm_kw),
                        run_arm(part, base_dir / "arms" / arm, base_agent, config, procs, ascension, resume=True, **arm_kw))

            smoke_n = int((before.get("sequential") or {}).get("smoke") or SEQ_SMOKE)
            batch_n = int((before.get("sequential") or {}).get("batch") or SEQ_BATCH)

            def walk_verdict(cr: list[dict], br: list[dict]) -> dict[str, Any]:
                kv = keep_v2_verdict(cr, br, rows, cand_gen=cand_agent.generation, cand_sha=cand_sha, block=name, frontier_info=frontier_info, boot=boot,
                                     apply_cap=False, head_sha=head_sha, recipe=recipe, rule=rule, final=False, min_gain=gain, **s1_kw)
                return compare_arms(cr, br, ALPHA, boot, None, keep=kv)

            walked = funnel(seeds, rows_at, walk_verdict, smoke=smoke_n, batch=batch_n, keep_rule="v3", frontier=frontier_info["index"],
                            frontier_name=frontier_info["name"], extend_possible=False, min_gain=gain, route_s=True, say=lambda m: print(m, flush=True))
            cand_rows, base_rows = walked["cand_rows"], walked["base_rows"]
            seq_doc = sequential_doc(walked, seeds, smoke=smoke_n, batch=batch_n, played=walked["stop"]["n"], fut_min=None, v2=True, rule=rule,
                                     frontier_name=frontier_info["name"], route_s=True)
            if walked["stop"]["action"] == "final":
                stop, played = None, len(seeds)
            else:
                stop, played = {"action": walked["stop"]["action"], "played": walked["stop"]["n"], "block_seeds": len(seeds), "sequential": seq_doc}, walked["stop"]["n"]
        keep = keep_v2_verdict(cand_rows, base_rows, rows, cand_gen=cand_agent.generation, cand_sha=cand_sha, block=name,
                               frontier_info=frontier_info, boot=boot, apply_cap=False, head_sha=head_sha, recipe=recipe,
                               rule=rule, pooled=list(pool_blocks), final=stop is None, min_gain=gain, **s1_kw)
    res = compare_arms(cand_rows, base_rows, ALPHA, boot, guard, **({"keep": keep} if v2 else {}))
    if res["paired_seeds"] != played + pre_n:
        raise ValueError(f"only {res['paired_seeds']} of {played + pre_n} seeds are paired in both cached arms")
    if stop is not None:  # the funnel's own check joins the others, as `promote` writes it for the same stop
        d = res["decision"]
        d["checks"]["sequential"] = stop["action"] not in STOP_FAILS
        d["promote"] = checks_hold(d["checks"], d.get("advisory_checks"))
        d["rule"] = f"{d['rule']}+{SEQ_RULE}" + (f"+{futility_rule(played)}" if stop["action"] == "futility" else "")
    if before is not None and out is None:  # the earlier record stays beside the new one, under a name not yet taken (a second rejudge of a day)
        stem = f"promotion-{before.get('decision_text', 'earlier')}-{str(before.get('written', 'x'))[:10]}"
        kept, j = Path(cand) / f"{stem}.json", 1
        while kept.exists():
            j += 1
            kept = Path(cand) / f"{stem}-r{j}.json"
        old.rename(kept)
    doc = {"written": utc_now(), "seed_block": name, "n_seeds": played + pre_n, "config": config, "alpha": ALPHA,
           "cand": {"agent": cand_agent.describe(), "summary": arm_summary(cand_rows)},
           "base": {"agent": base_agent.describe(), "summary": arm_summary(base_rows)}, **res,
           "decision_text": decision_text(res["decision"]), "cand_sha": cand_sha, "run_head_sha": head_sha,
           **({"macro": recipe} if recipe else {}), "rejudged": {"note": note, "earlier_decision": (before or {}).get("decision_text")},
           **({"keep_rule": rule} if v2 else {}), "f7": "not run", "matched_wall_clock": False}
    if pool_blocks:
        doc["pooled"] = pooled_record(name, played, own, pool_info)
    if route_s_mode is not None:  # route S: what was asked, the reading, how far the whole run moved from it, the check on the champion's stored arm
        doc["route_s_mode"] = route_s_mode
        if s1 is not None:
            doc["s1"] = s1
        shift = ((keep or {}).get("route_s") or {}).get("state_shift")
        if shift:
            doc["state_shift"] = shift
        if reuse is not None:
            doc["base_reuse"] = reuse
    if walked is not None:  # the block was completed: S2's walk is the record, and the rejudgement says what it added
        doc["rejudged"]["completed"] = {"from_seeds": earlier_played, "to_seeds": played, "block": len(seeds), "earlier_stop": earlier_stop,
                                        "text": f"completed from {earlier_played} to {played} of {len(seeds)} seeds under route S (the stored rows kept)",
                                        "row_backups": backups}
        if stop is None:
            doc["sequential"] = seq_doc
    cells = keep_cells(keep, attempt) if v2 else guard_cells(guard, attempt)
    if pool_blocks:
        cells["guard"] = f"{cells['guard']} | {pooled_say(doc['pooled'])}".replace("\t", " ")
    if route_s_note(route_s_mode, s1):
        cells["guard"] = f"{cells['guard']} | {route_s_note(route_s_mode, s1)}".replace("\t", " ")
    if stop is not None:
        said = f"{played} of {len(seeds)} seeds, where the funnel stopped on a {stop['action']}"
        doc["rejudged"]["seeds"] = {"judged": played, "block": len(seeds), "funnel_stop": stop["action"],
                                    "text": f"judged the {said}; the arms hold no others"}
        doc["sequential"] = {**stop["sequential"], "note": f"rejudged on the {said}: floor, boss_kill and wins are the figures at the stop"}
        said_row = (f"funnel {futility_rule(played)} of {len(seeds)} seeds" if stop["action"] == "futility"
                    else f"funnel {stop['action']} at {played} of {len(seeds)} seeds")
        cells["guard"] = f"{cells['guard']} | {said_row}".replace("\t", " ")
    dest = Path(cand) if out is None else Path(out)  # a dry run writes under `out`: a copy of the ledger, never the agent folder
    if out is not None:
        dest.mkdir(parents=True, exist_ok=True)
        ledger = dest / "promotions.tsv"
        if (Path(agent_root) / "promotions.tsv").is_file():
            ledger.write_bytes((Path(agent_root) / "promotions.tsv").read_bytes())
        else:
            ledger.unlink(missing_ok=True)
    if v2:
        doc["human_bar"] = human_bar_note(cand_agent, base_agent)  # report-only: no rule reads it
    write_json(dest / "promotion.json", doc)
    f, bk = doc["floor"], doc["boss_kill"]
    append_promotion(ledger,
                     {"written": doc["written"], "cand_gen": cand_agent.generation, "base_gen": base_agent.generation,
                      "seed_block": name, "seeds": doc["n_seeds"], "d_floor": f"{f['mean']:.4f}", "floor_lower": f"{f['lower']:.4f}",
                      "p_floor": f"{f['p']:.5f}", "boss_cand": bk["cand"], "boss_base": bk["base"], "p_boss": f"{bk['p']:.5f}",
                      "wins_cand": doc["cand"]["summary"]["wins"], "wins_base": doc["base"]["summary"]["wins"],
                      "decision": doc["decision_text"], "cand_sha": cand_sha, "cand_dir": Path(cand).name,
                      "run_head_sha": head_sha, "macro": recipe, **cells})
    if res["decision"]["promote"]:
        write_json((Path(agent_root) if out is None else dest) / "current.json",
                   {"generation": cand_agent.generation, "dir": str(Path(cand).resolve()), "promoted": doc["written"], "seed_block": name, "note": note})
        if lab is not None:
            p = Path(lab) / "score" / "pending"
            p.parent.mkdir(parents=True, exist_ok=True)
            p.write_text(utc_now() + "\n", encoding="utf-8")
    return doc


def resolve_e_block(template: str, k: int, n: int = 2000, *, tsv: Path | None = None, cand_sha: str | None = None,
                    also_dirs: Sequence[Path | str] | None = None, head_sha: str = "", recipe: str = "") -> Path:
    """The E-block for candidate generation `k`: `template` with {k} for the generation and {date} for today (UTC).
    An existing `E<k>-*.json` that has not judged another candidate (per `promotions.tsv`) is used as it is
    (E1-2026-10-23 is committed). If every such block did, a new `E<k>-<date>-r<n>` is created by the seed rule, so a
    candidate is never judged on a block that already judged a different one; a candidate is its combat prior and its run head
    (`candidate_identity`, `head_sha`) under a chooser recipe (`candidate_recipe`), so a second attempt that keeps the prior and changes
    the head, or the same pair under another recipe, gets a block of its own. A new block is written to the template's
    folder (the lab's `seeds` folder, outside the checkout); existing blocks are looked for there and in `also_dirs`
    (default: the checkout's committed folder), and a new name is free in all of them."""
    import gen0  # noqa: PLC0415

    path = Path(template.format(k=k, date=datetime.now(UTC).strftime("%Y-%m-%d")))
    found: dict[str, Path] = {}
    for d in (path.parent, *map(Path, [COMMITTED_SEEDS] if also_dirs is None else also_dirs)):
        for p in sorted(d.glob(f"E{k}-*.json")):
            if not p.stem.endswith(BASE_BLOCK_SUFFIX):  # a base block (the champion alone, no candidate) is no block to judge on
                found.setdefault(p.stem, p)
    existing = [found[name] for name in sorted(found)]
    for p in existing:
        if not block_used_by_another(tsv, p.stem, cand_sha or "none", head_sha, recipe):
            return p
    if existing:
        j = 1
        while f"{path.stem}-r{j}" in found or (path.parent / f"{path.stem}-r{j}.json").exists():
            j += 1
        path = path.parent / f"{path.stem}-r{j}.json"
    gen0.write_seed_file(path.stem, n, path)
    return path


def extension_of(cand: Path) -> dict[str, Any] | None:
    """The candidate's promotion.json when the keep rule v3 extended it (its decision says `extend` and promotes nothing), else None: the
    train job's cue to play the candidate on a second fresh block and judge both pooled."""
    p = Path(cand) / "promotion.json"
    if not p.is_file():
        return None
    doc = json.loads(p.read_text(encoding="utf-8"))
    d = doc.get("decision") or {}
    return doc if d.get("extend") and not d.get("promote") else None


def next_extension_block(first: Path, tsv: Path | None = None, *, n: int | None = None) -> Path:
    """The fresh E-block of an extension, written beside `first` (the block that extended the candidate): the date's next `-r<j>` name
    (`E18-2026-11-18-r1` -> `E18-2026-11-18-r2`), the first `j` that is no file in the folder or the checkout's committed blocks and no block of
    `tsv`'s rows, with as many seeds as `first` (`n` when named). The seeds derive from the name, so they are fresh and disjoint as every
    E-block's are (`promote` checks that again). An existing file is never reused: it may have judged another candidate."""
    import gen0  # noqa: PLC0415

    first = Path(first)
    stem = re.sub(r"-r\d+$", "", first.stem)
    judged = {r.get("seed_block") for r in read_promotions(tsv)}
    j = 1
    while (first.parent / f"{stem}-r{j}.json").exists() or (COMMITTED_SEEDS / f"{stem}-r{j}.json").exists() or f"{stem}-r{j}" in judged:
        j += 1
    path = first.parent / f"{stem}-r{j}.json"
    gen0.write_seed_file(path.stem, int(n or len(gen0.load_seeds(first))), path)
    return path


def run_promote_subprocess(emu_python: str, cand: Path, base: Path | None, seeds_file: Path, agent_root: Path, procs: int,
                           lib: str | None, lab: Path | None, lab_runs: Path | None = None, review_guard: bool = False,
                           prior_server_flags: tuple[str, ...] = (), sequential: bool = False, keep_rule: str | None = None,
                           futility: bool | None = None, futility_min_seeds: int | None = None, screen: bool | None = None,
                           pool_blocks: Sequence[str] = (), keep_rule_why: str | None = None, min_gain: float | None = None,
                           route_s: str | None = None, base_block: str | None = None) -> int:
    """From the train job (GPU venv): promotion plays emulators, which run in the emulator venv. The child inherits the
    environment, so the SPIRELINE_REVIEW_GUARD variable reaches it; `review_guard` passes the explicit flag on, and
    `prior_server_flags` the server options (none by default: the arms play on the CPU path); `sequential` plays the block as the
    funnel (`promote(sequential=True)`); `keep_rule` ("v1", "v2" or "v3") is passed on as `--keep-rule` when named, and the child reads the
    SPIRELINE_KEEP_RULE variable otherwise; `pool_blocks` (v3) are passed on as `--pool-block`, one each; `keep_rule_why` (the train job's choice
    between the marker's rule and the lab's) as `--keep-rule-why`; `min_gain` as `--min-gain` when it is named and is not the default (+1.0);
    `route_s` as `--route-s auto` when it is "auto" (the child's default is off: nothing is passed for off); `base_block` as `--base-block off` when it is "off"
    (the child's default is auto, which only acts under route S: nothing is passed for auto)."""
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
    if sequential:
        cmd += ["--sequential"]
    if keep_rule:
        cmd += ["--keep-rule", keep_rule]
    if keep_rule_why:
        cmd += ["--keep-rule-why", keep_rule_why]
    if min_gain is not None and float(min_gain) != MIN_GAIN_DEFAULT:  # the child's default is +1.0; only another minimum is passed on
        cmd += ["--min-gain", f"{float(min_gain):g}"]
    if futility is False:  # the child's default is on; only a switch-off or another start is passed on
        cmd += ["--no-futility"]
    if futility_min_seeds:
        cmd += ["--futility-min-seeds", str(int(futility_min_seeds))]
    if screen is False:
        cmd += ["--no-screen"]
    if route_s_of(route_s) == "auto":
        cmd += ["--route-s", "auto"]
    if base_block_of(base_block) == "off":
        cmd += ["--base-block", "off"]
    for block in pool_blocks:
        cmd += ["--pool-block", str(block)]
    cmd += list(prior_server_flags)  # none by default: the promotion arms play on the CPU path unless a flag says otherwise
    return subprocess.run(cmd, check=False).returncode


__all__ = [n for n in dir() if not n.startswith("_")]
_ = dataclasses
