#!/usr/bin/env python3
"""The round's job chain (memo row 5; docs/rl/ROUND.md): play, label, refit-prior, refit-judgement, candidate, smoke, block.

    python rl/round1.py play|label|refit-prior|refit-judgement|candidate|smoke|block --round-dir DIR [--config FILE] [--out JOBDIR] [--dry-run]
    python rl/round1.py fragment --round-dir DIR [--config FILE] --out jobs-fragment.json
    python rl/round1.py dry-run --scratch DIR            # the whole chain on a tiny sample in a scratch folder (nothing real is touched)
    python rl/round1.py config [--write FILE]

Every step reads the files of the steps before it from the round folder and writes its own (table in ROUND.md); `chain.json` records each
step's inputs and outputs, every one with its sha256 and size (an input also names the step that left it), and a step refuses to start on an
input that no longer has the sha256 its producer recorded. Every weight, lambda and strength is a key of the config file (`round1_data.DEFAULTS`). `--dry-run` runs the
step on the tiny sample of `TINY` in `<round-dir>/dry/` and starts nothing real: no block, no real game, no write under the agent or lab folders.
"""

from __future__ import annotations

import argparse
import datetime as dt
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import round1_data as rd  # noqa: E402

STEPS = ("play", "label", "refit-prior", "refit-judgement", "candidate", "smoke", "block")
# what a step needs from the folder and what it leaves there (relative to the round folder): the file handoffs
IO = {
    "play": ([], ["play/play.json"]),
    "label": (["play/play.json"], ["labels/labels.jsonl", "labels/manifest.json", "labels/label-summary.json"]),
    "refit-prior": (["play/play.json", "labels/labels.jsonl"], ["prior/weights.npz", "prior/refit.json"]),
    "refit-judgement": (["play/play.json", "labels/labels.jsonl", "prior/weights.npz"],
                        ["judgement/run_value.npz", "judgement/vrun_fit.json", "judgement/refit.json"]),
    "candidate": (["prior/weights.npz", "prior/refit.json", "judgement/vrun_fit.json", "judgement/run_value.npz"], ["candidate/candidate.json"]),
    "smoke": (["candidate/candidate.json"], ["smoke/smoke.json"]),
    "block": (["candidate/candidate.json", "smoke/smoke.json"], ["block/block.json"]),
}
PRODUCER = {rel: step for step, (_needs, leaves) in IO.items() for rel in leaves}  # the step that leaves each file
CANDIDATE_SKIP = ("promotion.json",)  # what a judgement writes into the candidate folder (its arms/ is a folder)
# the tiny sample of a --dry-run (docs/rl/ROUND.md): two runs under a 4-simulation search, K 1, one human shard, one epoch
TINY = {"play": {"runs": 2, "config": "flat-4", "procs": 2}, "label": {"k": 1, "procs": 1, "per_run": 1, "per_act": 0, "max_options": 3,
                                                                        "quota": {"card/a1": 2}, "heldout_fraction": 0.5, "act2_min_share": 0.0},
        "prior": {"epochs": 1, "batch": 64, "max_agent_decisions": 200, "label_gap_se": 0.0},
        "judgement": {"human_weights": [1.0], "seeds": [1], "epochs": 1, "batch": 256, "ranking_check": False, "device": "cpu"}}


def utc() -> str:
    return dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    tmp.replace(path)


def read_json(path: Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha256(path: Path | str) -> str:
    import hashlib  # noqa: PLC0415

    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def file_record(path: Path | str) -> dict:
    """A handoff file as chain.json keeps it: its sha256 and size."""
    return {"sha256": sha256(path), "bytes": Path(path).stat().st_size}


def folder_files(folder: Path | str, skip: tuple[str, ...] = CANDIDATE_SKIP) -> dict[str, dict]:
    """The files of a folder (top level) with their sha256 and size: a built candidate's identity, checked again at the smoke and the block."""
    return {f.name: file_record(f) for f in sorted(Path(folder).iterdir()) if f.is_file() and f.name not in skip}


def config_sha256(cfg: dict) -> str:
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(json.dumps(cfg, sort_keys=True).encode("utf-8")).hexdigest()


class Ctx:
    """The config, the round folder and (for a dry run) the scratch agent and lab it stands in."""

    def __init__(self, cfg: dict, rdir: Path, dry: bool = False, agent_root: Path | None = None, lab_runs: Path | None = None):
        self.dry = dry
        self.cfg = rd._merge(cfg, TINY) if dry else cfg  # noqa: SLF001
        self.dir = Path(rdir) / "dry" if dry and rdir.name != "dry" else Path(rdir)
        self.agent_root = Path(agent_root or self.cfg["paths"]["agent_root"])
        self.lab_runs = Path(lab_runs or self.cfg["paths"]["lab_runs"])
        self.lib = Path(self.cfg["paths"]["lab_tree"]) / "out"

    def p(self, rel: str) -> Path:
        return self.dir / rel

    def chain(self) -> dict:
        return read_json(self.p("chain.json")) if self.p("chain.json").is_file() else {"steps": {}}

    def need(self, step: str) -> None:
        """Refuse to start without the step's inputs, or on an input whose sha256 is no longer the one its producer recorded."""
        missing = [r for r in IO[step][0] if not self.p(r).is_file()]
        if missing:
            raise SystemExit(f"round1 {step}: missing {missing} in {self.dir} (run the step before it)")
        steps = self.chain()["steps"]

        def recorded(rel: str) -> str | None:  # an entry written before the files carried sizes is the bare hash
            got = (steps.get(PRODUCER[rel]) or {}).get("outputs", {}).get(rel)
            return got.get("sha256") if isinstance(got, dict) else got

        changed = [r for r in IO[step][0] if recorded(r) not in (None, sha256(self.p(r)))]
        if changed:
            raise SystemExit(f"round1 {step}: {changed} changed since {sorted({PRODUCER[r] for r in changed})} recorded them in chain.json "
                             f"(run that step again, or restore the file)")

    def done(self, step: str, extra: dict | None = None, artifacts: dict | None = None) -> dict:
        """Record the step in chain.json: every input and output file with its sha256 and size (an input names the step that left it),
        `sha256` the first output's, the config's sha256, and `artifacts` (files outside the round folder the step produced or read,
        e.g. the candidate folder's files)."""
        outs = {r: (self.p(r).is_file() and file_record(self.p(r))) for r in IO[step][1]}
        if not all(outs.values()):
            raise SystemExit(f"round1 {step}: did not leave {[r for r, v in outs.items() if not v]}")
        chain = self.chain()
        entry = {"at": utc(), "config_sha256": config_sha256(self.cfg),
                 "inputs": {r: {**file_record(self.p(r)), "from": PRODUCER.get(r)} for r in IO[step][0]}, "outputs": outs,
                 "sha256": outs[IO[step][1][0]]["sha256"], **({"artifacts": artifacts} if artifacts else {}), **(extra or {})}
        chain["steps"][step] = entry
        write_json(self.p("chain.json"), chain)
        return entry

    def agent(self) -> Any:
        import gen_loop  # noqa: PLC0415

        return gen_loop.load_agent(self.agent_root / "current.json")


# --------------------------------------------------------------------------------------------------------- play


def play_runs(ctx: Ctx) -> tuple[Any, str, list[dict]]:
    import hindsight  # noqa: PLC0415

    agent = ctx.agent()
    name = agent.config_name(ctx.cfg["play"]["config"])
    runs = hindsight.exploring_runs(ctx.lab_runs, agent.generation, agent.describe()["prior_sha256"], config=name, macro=agent.macro,
                                    stakes=agent.stakes)
    return agent, name, runs


def step_play(ctx: Ctx) -> dict:
    """Index the exploring runs of the current player (macro, stakes schedule, combat prior) in the lab's runs folder and, if fewer than
    play.runs exist, play the shortfall (gen0.py emu --explore-tau ... --record) into a new job folder with its job.json."""
    P = ctx.cfg["play"]
    agent, name, runs = play_runs(ctx)
    short = max(0, int(P["runs"]) - len(runs))
    played = None
    if short:
        job = ctx.lab_runs / (dt.datetime.now(dt.timezone.utc).strftime("%Y%m%dT%H%M") + "-selfplay")
        cmd = [sys.executable, str(ROOT / "rl" / "gen0.py"), "emu", "--stream", job.name, "--n", str(short), "--config", P["config"],
               "--procs", str(P["procs"]), "--lib", str(ctx.lib), "--agent-root", str(ctx.agent_root), "--record", "--out", str(job),
               "--explore-tau", str(P["explore_tau"]), "--explore-eps", str(P["explore_eps"])]
        if not ctx.dry:
            cmd += ["--prior-server", "auto", "--gpu-python", ctx.cfg["paths"]["gpu_python"]]
        job.mkdir(parents=True, exist_ok=True)
        t0 = time.time()
        rc = subprocess.run(cmd, cwd=str(ROOT), env={**os.environ, "STS2_LIB_PATH": str(ctx.lib), "STS2_ALLOW_STALE_NATIVE": "1"}).returncode
        write_json(job / "job.json", {"name": job.name, "params": {"cmd": cmd}, "rc": rc, "note": "written by rl/round1.py play"})
        played = {"job": job.name, "runs": short, "rc": rc, "wall_s": round(time.time() - t0, 1)}
        if rc:
            raise SystemExit(f"round1 play: gen0.py emu failed (rc {rc})")
        agent, name, runs = play_runs(ctx)
    doc = {"round": ctx.cfg["round"], "created": utc(), "agent": agent.describe(), "config": name, "wanted": int(P["runs"]),
           "runs": len(runs), "jobs": sorted({r["_job"] for r in runs}), "played_now": played}
    write_json(ctx.p("play/play.json"), doc)
    return ctx.done("play", {"runs": len(runs)})


# --------------------------------------------------------------------------------------------------------- label


def step_label(ctx: Ctx) -> dict:
    import hindsight  # noqa: PLC0415

    ctx.need("label")
    L = ctx.cfg["label"]
    agent, name, runs = play_runs(ctx)
    out = ctx.p("labels")
    quota = ",".join(f"{k}={v}" for k, v in L["quota"].items())
    ns = argparse.Namespace(out=str(out), agent_root=str(ctx.agent_root), agent_dir=str(Path(agent.path).relative_to(ctx.agent_root).as_posix()),
                            generation=agent.generation, runs_root=str(ctx.lab_runs), config=ctx.cfg["play"]["config"], k=int(L["k"]),
                            procs=int(L["procs"]), per_run=int(L["per_run"]), per_act=int(L["per_act"]), max_options=int(L["max_options"]),
                            quota=quota, quota_per_cell=0,
                            limit_runs=0, order_seed=int(L["order_seed"]), task_cap_s=float(L["task_cap_s"]), until="", judge="", doors=False,
                            lib=str(ctx.lib), no_hold=ctx.dry, own_schedule=bool(L["own_schedule"]), cont_boss_mult=int(L["cont_boss_mult"]),
                            run_config=name, macro=agent.macro, stakes=agent.stakes, prior_sha=agent.describe()["prior_sha256"],
                            rates=json.dumps({k: 1.0 for k in rd.LABEL_KINDS}) if ctx.dry else json.dumps(L["rates"]))
    rc = hindsight.pilot_cmd(ns)
    if rc:
        raise SystemExit(f"round1 label: the label run did not start (hindsight returned {rc})")
    consolidate_labels(out, L, name)
    return ctx.done("label", {"decisions": len(rd.label_records(out))})


def consolidate_labels(out: Path, L: dict, name: str) -> None:
    """labels.d/*.jsonl -> labels.jsonl (the documented file: usable records only, one per decision, schema, schedule, heldout flag) and
    label-summary.json."""
    recs = rd.label_records(out)
    sd = rd.pooled_pair_sd(recs)
    with open(out / "labels.jsonl.tmp", "w", encoding="utf-8") as fh:
        for r in recs:
            # schedule: the search config the exploring play (and its replay) ran under; cont_schedule: the one the continuations ran under
            # (a record written before the continuations had a schedule of their own played the same one)
            fh.write(json.dumps({**r, "schema": rd.LABEL_SCHEMA, "schedule": name, "cont_schedule": r.get("cont_schedule", name),
                                 "heldout": rd.heldout(r["run"]["seed"], L["heldout_fraction"])}) + "\n")
    (out / "labels.jsonl.tmp").replace(out / "labels.jsonl")
    cells: dict[str, int] = {}
    for r in recs:
        c = f"{r['kind']}/a{r['act'] + 1}"
        cells[c] = cells.get(c, 0) + 1
    by_act = {a: sum(n for c, n in cells.items() if c.endswith(f"/{a}")) for a in ("a1", "a2")}
    write_json(out / "label-summary.json", {"decisions": len(recs), "k": int(L["k"]), "schedule": name, "pooled_pair_sd": sd, "cells": cells,
                                            "by_act": {**by_act, "act2_share": (by_act["a2"] / len(recs)) if recs else None},
                                            "schedules": {"play": name, "continuations": sorted({r.get("cont_schedule", name) for r in recs})},
                                            "heldout": sum(rd.heldout(r["run"]["seed"], L["heldout_fraction"]) for r in recs),
                                            "schema": rd.LABEL_SCHEMA})


# --------------------------------------------------------------------------------------------------------- refit-prior


def judge_nets(ctx: Ctx, judge_npz: Path, prior: Path, lam: float) -> Any:
    import imitation_chooser as ic  # noqa: PLC0415

    return ic.BlendedPolicy(str(judge_npz), str(prior), lam, W=53.0).nets


def label_mix(recs: list[dict]) -> dict:
    """Label records by act and by kind and act: {"a1", "a2", "cells": {"card/a1": n ...}} (cells under 20 are kept: these are our own runs)."""
    cells: dict[str, int] = {}
    for r in recs:
        c = f"{r['kind']}/a{r['act'] + 1}"
        cells[c] = cells.get(c, 0) + 1
    return {"a1": sum(n for c, n in cells.items() if c.endswith("/a1")), "a2": sum(n for c, n in cells.items() if c.endswith("/a2")),
            "cells": dict(sorted(cells.items()))}


def step_refit_prior(ctx: Ctx) -> dict:
    import round1_prior as rp  # noqa: PLC0415

    ctx.need("refit-prior")
    cfg, P = ctx.cfg, ctx.cfg["prior"]
    agent = ctx.agent()
    play = read_json(ctx.p("play/play.json"))
    old = Path(cfg["paths"]["prior_weights"])
    nets = judge_nets(ctx, Path(agent.macro_npz), old, float(P["lambda"]))
    recs = rd.label_records(ctx.p("labels"))
    held = [r for r in recs if rd.heldout(r["run"]["seed"], cfg["label"]["heldout_fraction"])]
    train = [r for r in recs if r not in held]
    sd = rd.pooled_pair_sd(train)
    t_items, t_stats = rd.label_items(train, ctx.lab_runs, nets, winner_sd=sd, winner_z=float(P["label_gap_se"]))
    h_items, h_stats = rd.label_items(held, ctx.lab_runs, nets)
    wins = sum(it.winner is not None for it in t_items)
    jobs = play["jobs"]
    n = int(P["max_agent_decisions"])
    # a source with weight 0 is not trained on: its training decisions are not read (the held-out ones still are, for the report)
    a_train = (rd.agent_items(ctx.lab_runs, jobs, nets, keep=1.0, hold=False, fraction=cfg["label"]["heldout_fraction"], limit=n)
               if float(P["weights"]["agent"]) > 0 else [])
    lam_acts = {int(a): float(v) for a, v in P["lambda_acts"].items()}
    a_held = rd.agent_items(ctx.lab_runs, jobs, nets, keep=1.0, hold=True, fraction=cfg["label"]["heldout_fraction"], limit=max(1, n // 4))
    human_dir = Path(cfg["paths"]["human_prior_dir"]) / cfg["paths"]["human_decisions"]
    if ctx.dry:  # one human shard, hard-linked into the scratch folder
        small = ctx.p("human-shard")
        small.mkdir(parents=True, exist_ok=True)
        shards = sorted(human_dir.glob("shard-*.npz"))[:1]
        for s in shards:
            if not (small / s.name).exists():
                os.link(s, small / s.name) if os.name != "nt" else shutil.copy2(s, small / s.name)
        shutil.copy2(human_dir / "counts.json", small / "counts.json")
        human_dir = small
    train_sets = {"agent": rp.sets_from_items(a_train), "label": rp.sets_from_items(t_items, label_winners=True)}
    val_sets = {"agent": rp.sets_from_items(a_held), "label": rp.sets_from_items(h_items)}
    res = rp.fit(cfg, human_dir=human_dir, train_sets=train_sets, val_sets=val_sets, old_path=old, out_dir=ctx.p("prior"),
                 dev_name="cpu" if ctx.dry else None)
    leg = rp.label_leg_report(h_items, nets, old, Path(res["weights"]), float(P["lambda"]), list(P["lambda_grid"]), lam_acts or None)
    lam, why = rp.choose_lambda(cfg, leg)
    report = {"round": cfg["round"], "written": utc(), "config": P, "weights": res["weights"], "sha256": res["sha256"], "old_prior": str(old),
              "old_prior_sha256": sha256(old), "lambda": lam, "lambda_acts": {str(a): v for a, v in sorted(lam_acts.items())},
              "lambda_reason": why, "attribution": rp.SPIRE_CODEX,
              "labels": {"records": len(recs), "train": len(train), "heldout": len(held), "pooled_pair_sd": sd, "clear_winners": wins,
                         "rebuild_train": t_stats, "rebuild_heldout": h_stats},
              "agent_choices": {"train": len(a_train), "heldout": len(a_held)}, "fit": {k: v for k, v in res.items() if k != "weights"},
              "label_mix": {"train": label_mix(train), "heldout": label_mix(held)},
              "label_leg": leg, "meaning": "label_leg: mean centred label value A (floors) of each arm's pick among the played options on "
                                         "held-out runs; *_minus_old_blend is the paired difference from the current blend"}
    write_json(ctx.p("prior/refit.json"), report)
    return ctx.done("refit-prior", {"sha256": res["sha256"], "lambda": lam})


# --------------------------------------------------------------------------------------------------------- refit-judgement


def step_refit_judgement(ctx: Ctx) -> dict:
    import round1_judgement as rj  # noqa: PLC0415
    import round1_prior as rp  # noqa: PLC0415
    import gen_loop  # noqa: PLC0415
    from run_train import keep, models  # noqa: PLC0415
    from run_train import train as rt  # noqa: PLC0415
    from run_train.engine import CombatBudget  # noqa: PLC0415

    ctx.need("refit-judgement")
    cfg, J = ctx.cfg, ctx.cfg["judgement"]
    agent = ctx.agent()
    play = read_json(ctx.p("play/play.json"))
    sha = agent.describe()["prior_sha256"]
    mans = [(d, m) for d, m in gen_loop.fighter_manifests(ctx.lab_runs, sha, None, exploring_only=True)
            if d.parent.name in set(play["jobs"]) and ((m.get("agent") or {}).get("stakes") == agent.stakes)]
    if not mans:
        raise SystemExit("round1 refit-judgement: no recorded exploring blocks of the current player in the play index")
    configs = {(int(m["base_sims"]), int(m["ascension"])) for _, m in mans}
    if len(configs) != 1:
        raise SystemExit(f"round1 refit-judgement: the window mixes budgets {sorted(configs)}")
    sims, asc = next(iter(configs))
    gens = sorted({int(m["generation"]) for _, m in mans})
    paths = [d / n for d, m in mans for n in m.get("run_shards", [])]
    tc = rt.TrainConfig(beta=float(J["beta"]), validation=float(J["heldout_fraction"]), threads=int(J["threads"]), rows_per_run=0,
                        value_generations=gens[-1] - gens[0] + 1)
    _fights, values = rt.load_buffers(paths, generation=gens[-1], combat_budget=CombatBudget(sims, sims), ascension=asc, config=tc)
    inc_pt = Path(agent.path) / "run_value.pt"
    fnet, vnet = models.FightNet(), models.RunValueNet(256, 0.1)
    models.load_torch(inc_pt, fnet, vnet)
    human_dir = Path(cfg["paths"]["human_prior_dir"]) / cfg["paths"]["human_decisions"]
    if ctx.dry:
        human_dir = ctx.p("human-shard")
        if not human_dir.is_dir():
            human_dir.mkdir(parents=True)
            src = Path(cfg["paths"]["human_prior_dir"]) / cfg["paths"]["human_decisions"]
            shutil.copy2(sorted(src.glob("shard-*.npz"))[0], human_dir)
            shutil.copy2(src / "counts.json", human_dir)
    hum = rj.human_rows(human_dir, list(cfg["prior"]["human_fifths_train"]), list(J["human_kinds"]))
    meta = {"generation": gens[-1], "era": {"combat_budget": str(CombatBudget.of(sims)), "ascension": asc}, "warm_start": str(inc_pt)}
    res = rj.refit(cfg, values=values, fight_net=fnet, value_net_state=vnet.state_dict(), hum=hum, out_dir=ctx.p("judgement"), meta=meta)
    out = ctx.p("judgement")
    keep_doc = keep.evaluate_value_net(values, res["value"])
    window_seeds = [str(x) for d, m in mans for name in m.get("combat_parts", []) for x in gen_loop.read_part(d / name)["seeds"]]
    if J["ranking_check"] and not ctx.dry:
        keep_doc["ranking"] = gen_loop.ranking_block(out / "run_value.npz", agent.macro_npz, ctx.lab_runs, out, exclude_seeds=window_seeds)
    # the label leg: held-out labels, the new judgement against the incumbent's, with the incumbent prior and the new prior (the label bound reads it)
    recs = [r for r in rd.label_records(ctx.p("labels")) if rd.heldout(r["run"]["seed"], cfg["label"]["heldout_fraction"])]
    prior_ref = read_json(ctx.p("prior/refit.json"))
    lam = float(prior_ref["lambda"])
    lam_acts = {int(a): float(v) for a, v in (prior_ref.get("lambda_acts") or {}).items()}
    new_lam: Any = (lambda it: float(lam_acts.get(it.act + 1, lam))) if lam_acts else lam
    old_prior, new_prior = Path(cfg["paths"]["prior_weights"]), ctx.p("prior/weights.npz")
    nets_old, nets_new = judge_nets(ctx, Path(agent.macro_npz), old_prior, lam), judge_nets(ctx, out / "run_value.npz", new_prior, lam)
    items, stats = rd.label_items(recs, ctx.lab_runs, nets_old)
    from human_prior_infer import PriorScorer  # noqa: PLC0415

    po, pn = PriorScorer.load(old_prior), PriorScorer.load(new_prior)
    v_old = [list(models.expected_floors(nets_old.value.floor_probs(i.rows), 53.0)) for i in items]
    v_new = [list(models.expected_floors(nets_new.value.floor_probs(i.rows), 53.0)) for i in items]
    lo, ln = [list(po.row_logits(i.rows)) for i in items], [list(pn.row_logits(i.rows)) for i in items]
    leg = rd.leg_picks(items, {"incumbent_blend": (v_old, lo, lam), "new_judgement_old_prior": (v_new, lo, lam),
                               "new_judgement_new_prior": (v_new, ln, new_lam), "new_judgement_alone": (v_new, ln, 0.0)}) if items else {}
    # D64: the export decides `kept`; keep test (1), D33's out-of-fold test of the fit, is recorded beside it, and (2) the ranking check with its
    # skip guard and (3) the label bound are reported too (judgement.guards "report", the default after the labels-against-blocks memo); only an
    # explicit "gate" in the round's config lets the guards decide
    guards = rj.guards_report(keep_doc, leg, J)
    gen_loop.keep_advisory(keep_doc)  # D64: the D33 skill and the ranking check are recorded, not gating
    core = (out / "run_value.npz").is_file()
    kept = (core and (guards["pass"] if guards["decides_kept"] else True)) or bool(ctx.dry and "ranking" not in keep_doc)  # a dry run carries the fit on: it proves the handoff, not the test
    write_json(out / gen_loop.VRUN_FIT_JSON, {
        "recipe": "round1: the incumbent's FightNet and run head refit on the current player's exploring runs (pure Monte Carlo, beta "
                  f"{J['beta']}) with human chosen-option rows (weight {res['human_weight']}) behind a source input; ROUND.md",
        "written": utc(), "prior_sha256": sha, "beta": J["beta"], "human_weight": res["human_weight"], "human_weight_grid": res["selection"]["grid"],
        "window": {"blocks": len(mans), "generations": gens, "exploring_only": True, "stakes": agent.stakes,
                   "runs_by_generation": {str(g): sum(int(m["runs"]) for _, m in mans if int(m["generation"]) == g) for g in gens},
                   "jobs": sorted(d.parent.name for d, _ in mans)},
        "keep": keep_doc, "guards": guards, "kept": kept, **({"dry_run": True} if ctx.dry else {}), "sha256": sha256(out / "run_value.npz")})
    report = {"round": cfg["round"], "written": utc(), "config": J, "sha256": sha256(out / "run_value.npz"), "window_runs": values.runs,
              "selection": res["selection"],
              "fit": {k: v for k, v in res.items() if k not in ("value",)}, "keep": {k: keep_doc.get(k) for k in ("status", "all_computable_pass")},
              "guards": guards, "kept": kept, "label_leg": leg, "label_rebuild": stats, "attribution": rp.SPIRE_CODEX,
              "meaning": "selection: the human weight chosen (rule weight_rule: the lowest held-out clears-Act-2 log-loss among the weights whose "
                         "clears-Act-1 log-loss is not worse than the incumbent's), the grid and each weight's figures; fit.grid: held-out self-play "
                         "log-loss of the clears-Act-1 and clears-Act-2 heads by human weight; kept_heldout is the average-of-seeds fit; guards: the "
                         "ranking check, the skip guard and the label bound as numbers (they decide `kept` only when guards.decides_kept); "
                         "label_leg: mean centred label value A of the blended picks, differences from the incumbent blend"}
    write_json(out / "refit.json", report)
    return ctx.done("refit-judgement", {"kept": kept, "human_weight": res["human_weight"]})


# --------------------------------------------------------------------------------------------------------- candidate, smoke, block


def promote_cmd(ctx: Ctx, which: str, cand: Path, seeds_file: str) -> list[str]:
    B = ctx.cfg[which]
    cmd = [sys.executable, str(ROOT / "rl" / "gen0.py"), "promote", "--cand", str(cand),
           "--seeds-file", seeds_file, "--agent-root", str(ctx.agent_root), "--procs", str(B["procs"]), "--config", "a3-v1", "--lib", str(ctx.lib),
           "--sequential", "--keep-rule", B["keep_rule"], "--smoke-seeds", str(B["seeds"] if which == "smoke" else ctx.cfg["smoke"]["seeds"])]
    if which == "block":
        cmd += ["--batch-seeds", str(B["batch_seeds"])]
        F = ctx.cfg["futility"]
        cmd += ["--futility-min-seeds", str(int(F["min_seeds"]))] + ([] if F["enabled"] else ["--no-futility"])
    return cmd + ["--prior-server", "auto", "--gpu-python", ctx.cfg["paths"]["gpu_python"], "--server-procs", "20", "--lab-runs", str(ctx.lab_runs)]


def step_candidate(ctx: Ctx) -> dict:
    """Build the candidate: the base's combat prior, the new judgement, the new prior at the chosen lambda, the current player's stakes
    schedule (gen_loop.build_candidate; the base is a plain generation, as the stakes candidates were built)."""
    import gen_loop  # noqa: PLC0415

    ctx.need("candidate")
    C = ctx.cfg["candidate"]
    agent = ctx.agent()
    meta = read_json(Path(agent.path) / "agent.json")
    base = ctx.agent_root / C["base"]
    ref = read_json(ctx.p("prior/refit.json"))
    sched = meta.get("search_schedule")
    tag = C["tag"] + ("-dry" if ctx.dry else "")
    cand = gen_loop.build_candidate(base, base, ctx.agent_root, int(C["target_generation"]), why=C["why"], export=not ctx.dry,
                                    vrun_fit=ctx.p("judgement"), tag=tag, human_prior=ctx.p("prior/weights.npz"), lam=float(ref["lambda"]),
                                    lam_acts={int(a): float(v) for a, v in (ref.get("lambda_acts") or {}).items()} or None,
                                    search_schedule={"base": sched["base"], "mult": sched["mult"]} if sched else None)
    files = folder_files(cand)
    marker_dir = (ctx.dir / "train") if ctx.dry else Path(ctx.cfg["paths"]["lab_runs"]).parent / "train"
    write_marker = bool(C["marker"]) and not ctx.dry  # the lab's train job would judge the candidate itself; the chain's smoke and block do
    marker = marker_dir / f"ready-{read_json(base / 'agent.json').get('generation', 11)}-g{C['target_generation']}-{tag}"
    write_json(ctx.p("candidate/candidate.json"), {"dir": str(cand), "tag": tag, "marker": str(marker), "marker_written": write_marker,
                                                    "lambda": ref["lambda"], "lambda_acts": ref.get("lambda_acts") or {}, "prior_sha256": ref["sha256"],
                                                    "recipe": gen_loop.candidate_recipe(gen_loop.load_agent(cand)), "files": files,
                                                    "judgement_sha256": sha256(ctx.p("judgement/run_value.npz")), "incumbent": str(agent.path)})
    if write_marker:
        marker_dir.mkdir(parents=True, exist_ok=True)
        marker.write_text(utc() + "\n", encoding="utf-8")
    return ctx.done("candidate", {"dir": str(cand)}, artifacts={f"candidate-folder/{n}": r for n, r in files.items()})


def step_judged(ctx: Ctx, which: str) -> dict:
    """smoke / block: the keep test v2's funnel (`gen0.py promote --sequential --keep-rule v2`) on the candidate against the current
    player. A dry run builds and checks the command against gen0's own parser and starts nothing."""
    import gen0  # noqa: PLC0415

    ctx.need(which)
    cdoc = read_json(ctx.p("candidate/candidate.json"))
    cand = Path(cdoc["dir"])
    files = folder_files(cand)
    if files != cdoc["files"]:
        raise SystemExit(f"round1 {which}: the candidate folder {cand.name} is not the one the candidate step recorded "
                         f"(changed: {sorted(n for n in set(files) | set(cdoc['files']) if files.get(n) != cdoc['files'].get(n))})")
    seeds = ctx.cfg[which]["seeds_file"].replace("{date}", dt.date.today().isoformat())
    cmd = promote_cmd(ctx, which, cand, seeds)
    args = gen0.build_parser().parse_args(cmd[2:])  # the command parses, flag for flag
    doc = {"command": [Path(cmd[1]).name, *cmd[2:]], "seeds_file": seeds, "seeds": ctx.cfg[which]["seeds"], "keep_rule": args.keep_rule,
           "incumbent": str(ctx.agent_root / "current.json"), "candidate": str(cand), "started": False}
    if ctx.dry:
        doc["note"] = "dry run: parsed, not started"
    else:
        if not Path(seeds).is_file():  # fresh seeds of their own name, checked disjoint from the lab's by promote (--lab-runs)
            subprocess.run([sys.executable, str(ROOT / "rl" / "gen0.py"), "seeds", "--name", Path(seeds).stem, "--n", str(ctx.cfg[which]["seeds"]),
                            "--out", seeds], cwd=str(ROOT), check=True)
        doc["started"], doc["rc"] = True, subprocess.run(cmd, cwd=str(ROOT)).returncode
    write_json(ctx.p(f"{which}/{which}.json"), doc)
    return ctx.done(which, {"started": doc["started"]}, artifacts={f"candidate-folder/{n}": r for n, r in files.items()})


STEP_FN = {"play": step_play, "label": step_label, "refit-prior": step_refit_prior, "refit-judgement": step_refit_judgement,
           "candidate": step_candidate, "smoke": lambda c: step_judged(c, "smoke"), "block": lambda c: step_judged(c, "block")}


# --------------------------------------------------------------------------------------------------------- the fragment and the dry run


def fragment(cfg: dict, rdir: Path, config_path: str) -> dict:
    """jobs.json entries for the chain (not installed). The cmd of each is `rl/round1.py <step>`; `needs` lists the files it reads."""
    py, gpu = "{python}", cfg["paths"]["gpu_python"]
    base = ["rl/round1.py"]
    rdir = Path(rdir)
    tail = ["--round-dir", rdir.as_posix(), "--config", config_path, "--out", "{job}"]
    out: dict[str, Any] = {"_note": "Fragment for LAB/jobs.json, NOT installed (D55 addendum 11, memo row 5). Keys are job kinds the runner does not "
                           "know yet (tools/lab/jobs.py NEW_KINDS); each runs rl/round1.py <step> and is resumable. Merge at a quiet time with "
                           "a jobs.json backup and a dated sentence in _note. Order: " + " -> ".join(f"round-{s}" for s in STEPS) + "."}
    for s in STEPS:
        py_ = gpu if s in ("refit-prior", "refit-judgement", "candidate") else py
        procs = {"play": 16, "label": cfg["label"]["procs"], "smoke": 16, "block": 16}.get(s, 4)
        out[f"round-{s}"] = {"enabled": False, "cmd": [py_, *base, s, *tail], "cwd": "{tree}", "timeout_min": {"play": 400, "label": 900, "block": 1200}.get(s, 180),
                             "procs": procs, "cores": procs, "gpu": s in ("refit-prior", "refit-judgement"), "after": f"round-{STEPS[STEPS.index(s) - 1]}" if s != "play" else None,
                             "needs": ["{tree}/rl/round1.py", *[(rdir / r).as_posix() for r in IO[s][0]]], "leaves": [(rdir / r).as_posix() for r in IO[s][1]]}
    return out


def scratch_agent(src_root: Path, dst: Path, base_search: str | None = "flat-4") -> Path:
    """A scratch agent folder for the dry run: the current player's files with its stakes schedule's base replaced by a 4-simulation search,
    plus its base generation (the candidate's base), so nothing under the real agent folder is touched."""
    import gen_loop  # noqa: PLC0415
    import macro_recipe  # noqa: PLC0415

    cur = gen_loop.load_agent(src_root / "current.json")
    meta = read_json(Path(cur.path) / "agent.json")
    gdir = dst / f"gen{cur.generation}" / Path(cur.path).name
    gdir.mkdir(parents=True, exist_ok=True)
    for f in Path(cur.path).iterdir():
        if f.is_file() and f.name not in ("promotion.json",) and f.suffix in (".npz", ".pt", ".json", ".onnx"):
            shutil.copy2(f, gdir / f.name)
    if base_search:  # a cheap search for the dry run; None keeps the player's own schedule (a fixture over recorded runs)
        mult = meta["search_schedule"]["mult"]
        tag = macro_recipe.stakes_tag(base_search, mult)
        meta["search_schedule"] = {"base": base_search, "mult": mult, "tag": tag}
        meta["recipe"] = macro_recipe.full_recipe(meta["macro"], tag, None)
        write_json(gdir / "agent.json", meta)
    base_rel = rd.DEFAULTS["candidate"]["base"]
    bdir = dst / base_rel
    bdir.mkdir(parents=True, exist_ok=True)
    for f in (src_root / base_rel).iterdir():
        if f.is_file() and f.suffix in (".npz", ".pt", ".json", ".onnx"):
            shutil.copy2(f, bdir / f.name)
    write_json(dst / "current.json", {"generation": cur.generation, "dir": str(gdir), "promoted": "scratch", "seed_block": "scratch"})
    return dst


def dry_run_chain(cfg: dict, scratch: Path, steps: tuple[str, ...] = STEPS) -> dict:
    """The whole chain on the tiny sample in `scratch`: a scratch agent folder and lab runs folder, every step's file handoff checked."""
    scratch.mkdir(parents=True, exist_ok=True)
    agent_root = scratch / "agent"
    if not (agent_root / "current.json").is_file():
        scratch_agent(Path(cfg["paths"]["agent_root"]), agent_root)
    lab = scratch / "lab" / "runs"
    lab.mkdir(parents=True, exist_ok=True)
    ctx = Ctx(cfg, scratch / "round", dry=True, agent_root=agent_root, lab_runs=lab)
    report = {}
    for s in steps:
        t0 = time.time()
        report[s] = {**STEP_FN[s](ctx), "wall_s": round(time.time() - t0, 1)}
        print(f"round1 dry-run: {s} ok ({report[s]['wall_s']} s)", flush=True)
    return report


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for s in STEPS:
        p = sub.add_parser(s)
        p.add_argument("--round-dir", required=True)
        p.add_argument("--config")
        p.add_argument("--out", help="the lab job folder: result.json goes there")
        p.add_argument("--dry-run", action="store_true", help="the tiny sample in <round-dir>/dry (needs a scratch agent: use the dry-run command)")
    fr = sub.add_parser("fragment")
    fr.add_argument("--round-dir", required=True)
    fr.add_argument("--config", default="")
    fr.add_argument("--out", required=True)
    dr = sub.add_parser("dry-run")
    dr.add_argument("--scratch", required=True)
    dr.add_argument("--config")
    dr.add_argument("--steps", default=",".join(STEPS))
    cf = sub.add_parser("config")
    cf.add_argument("--write")
    a = ap.parse_args(argv)
    if a.cmd == "config":
        if a.write:
            Path(a.write).write_bytes(rd.config_text().encode("utf-8"))
        else:
            print(rd.config_text(), end="")
        return 0
    cfg = rd.load_config(getattr(a, "config", None) or None)
    if a.cmd == "fragment":
        write_json(Path(a.out), fragment(cfg, Path(a.round_dir), a.config or "{tree}/rl/round1_config.json"))
        return 0
    if a.cmd == "dry-run":
        rep = dry_run_chain(cfg, Path(a.scratch), tuple(a.steps.split(",")))
        write_json(Path(a.scratch) / "dry-run.json", rep)
        return 0
    if a.dry_run:  # --round-dir is the scratch folder; earlier steps' tiny outputs must be there already
        rep = dry_run_chain(cfg, Path(a.round_dir), (a.cmd,))
        write_json(Path(a.round_dir) / f"dry-run-{a.cmd}.json", rep)
        return 0
    ctx = Ctx(cfg, Path(a.round_dir))
    res = STEP_FN[a.cmd](ctx)
    if a.out:
        write_json(Path(a.out) / "result.json", {"step": a.cmd, **res})
    return 0


if __name__ == "__main__":
    import round1 as _self  # noqa: PLC0415 -- the pool's workers unpickle `round1`, not `__main__`

    raise SystemExit(_self.main())
