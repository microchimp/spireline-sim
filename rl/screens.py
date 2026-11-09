"""Cheap screens before the judgement's big test (D55 addendum 18, 2026-11-09; docs/rl/KEEP_TEST_V2.md "The screens").

`gen_loop.promote` asks `kind_of(candidate folder, incumbent folder)` before it plays an arm. A candidate that differs from the incumbent
in one part only has a cheap test of that part:

  screen:fight  only the fight network differs (combat_policy.pt, combat_net.pt / .onnx): the candidate plays the fight exam's human stratum
                (400 starts, four draws; `fight_exam.py play --stratum human`, the machinery of the human-bar line) and is rejected when its
                survival is below the incumbent's, on the same starts and draws, by more than two standard errors (the bootstrap's over
                starts, within cells). The incumbent's side is the games of the gym player that holds its network bytes.
  screen:prior  only the human prior differs (human_prior.npz): both priors are scored on the held-out human choices of the prior's own fit
                (the validation fifth, never the exam fifth; `human_prior.score_total`, the summed mean negative log-likelihood of the kinds
                the two priors share) and the candidate is rejected when its log-loss is worse than the incumbent's.

Anything else (several parts differ, or only the settings, the run value or the stakes): no screen. A screen that cannot be read (no human
stratum, no games for the incumbent's network, engine tags that differ, a run that was blocked or stopped, the held-out decisions absent)
says why and the arms are played as before; a screen never fails a judgement. A rejection is a rejected judgement: `promotion.json` carries the
figures, the ledger row has seeds 0 and the rule `screen:fight` / `screen:prior`, and the attempt is spent as for any rejection.
"""

from __future__ import annotations

import hashlib
import json
import re
from collections.abc import Callable, Mapping
from pathlib import Path
from typing import Any

SCREEN_FIGHT = "screen:fight"
SCREEN_PRIOR = "screen:prior"
FIGHT_FILES = ("combat_policy.pt", "combat_net.pt", "combat_net.onnx")  # the network the fights are searched with
PRIOR_FILES = ("human_prior.npz",)
VALUE_FILES = ("run_value.npz", "run_value.pt", "run_value.onnx", "fight_net.pt", "fight_net.onnx")  # the run-level judgement (FightNet feeds it)
FIGHT_SE = 2.0  # screen:fight rejects below the incumbent's survival by more than this many standard errors
PI_H = re.compile(r"pi_h=[0-9a-f]{8}")  # the prior's tag inside a macro or recipe string


def sha256_file(path: Path | str) -> str | None:
    p = Path(path)
    if not p.is_file():
        return None
    h = hashlib.sha256()
    with p.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def settings_of(doc: Mapping[str, Any]) -> dict[str, Any]:
    """What an agent.json says about how the agent plays, apart from its weights: the macro and recipe (the prior's tag taken out: a different
    prior is a different file, compared as such), the mask version, the human prior's lambda, lambda by act and guard, the stakes schedule and
    the potion hold. Bookkeeping (generation, dates, lineage, notes, the fit's record) is left out."""
    hp = doc.get("human_prior") if isinstance(doc.get("human_prior"), dict) else {}
    return {"macro": PI_H.sub("pi_h=*", str(doc.get("macro"))), "recipe": PI_H.sub("pi_h=*", str(doc.get("recipe"))),
            "mask_version": doc.get("mask_version"), "lambda": hp.get("lambda"), "lambda_acts": hp.get("lambda_acts"), "guard": hp.get("guard"),
            "search_schedule": doc.get("search_schedule"), "potion_hold": doc.get("potion_hold")}


def parts(folder: Path | str) -> dict[str, Any]:
    """A candidate folder as the screens compare it: {"fight": {file: sha}, "prior": {...}, "value": {...}, "settings": {...}}."""
    d = Path(folder)
    try:
        doc = json.loads((d / "agent.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        doc = {}
    return {"fight": {f: sha256_file(d / f) for f in FIGHT_FILES}, "prior": {f: sha256_file(d / f) for f in PRIOR_FILES},
            "value": {f: sha256_file(d / f) for f in VALUE_FILES}, "settings": settings_of(doc)}


def differing(cand: Path | str, base: Path | str) -> list[str]:
    """The parts (fight, prior, value, settings) in which the candidate's folder differs from the incumbent's, in that order."""
    a, b = parts(cand), parts(base)
    return [k for k in ("fight", "prior", "value", "settings") if a[k] != b[k]]


def kind_of(cand: Path | str, base: Path | str) -> str | None:
    """"fight" when only the fight network differs, "prior" when only the human prior does, None otherwise (no screen)."""
    got = differing(cand, base)
    return got[0] if len(got) == 1 and got[0] in ("fight", "prior") else None


# ---------------------------------------------------------------------------------------------------------- screen:fight


def fight_reading(cand_games: dict, inc_games: dict, human: dict, *, boot: int = 400, se_bound: float = FIGHT_SE) -> dict:
    """The candidate's survival against the incumbent's on the human stratum, paired by start on the draws both played
    (`fight_exam.paired_starts`): in points per 100, the difference, its standard error (the bootstrap's over starts, resampled within their
    cells) and the verdict: `rejected` when the difference is below minus `se_bound` standard errors. Both sides' engine tags are carried:
    games on different engines are not compared (`comparable` false, never rejected)."""
    import fight_exam as fe  # noqa: PLC0415

    cand = {k: r for k, r in cand_games.items() if k[0] in human}
    inc = {k: r for k, r in inc_games.items() if k[0] in human}
    cand_engine, inc_engine = fe._tags(cand, "engine"), fe._tags(inc, "engine")  # noqa: SLF001
    out: dict[str, Any] = {"rule": SCREEN_FIGHT, "stratum_starts": len(human), "cand_games": len(cand), "inc_games": len(inc),
                           "cand_engine": cand_engine, "inc_engine": inc_engine, "comparable": cand_engine == inc_engine, "rejected": False}
    groups = fe.paired_starts(inc, cand, human, {sid: 1.0 for sid in human})
    items = [it for v in groups.values() for it in v]
    out["starts"] = len(items)
    if not items:
        return out
    b = fe.bootstrap(groups, fe.mean_stat("d_won"), boot=boot)
    cand_surv = 100 * fe.mean(it["cand_won"] for it in items)
    inc_surv = 100 * fe.mean(it["champ_won"] for it in items)
    diff = 100 * (b["estimate"] or 0.0)
    se = None if b["se"] is None else 100 * b["se"]
    out.update(cand_surv=round(cand_surv, 2), inc_surv=round(inc_surv, 2), diff=round(diff, 2), se=None if se is None else round(se, 3),
               z=None if not se else round(diff / se, 2), bound=se_bound)
    out["rejected"] = bool(out["comparable"] and se is not None and diff < -se_bound * se)
    return out


def fight_line(r: dict) -> str:
    if not r.get("starts"):
        return f"{SCREEN_FIGHT}: no paired starts"
    z = "n/a" if r.get("z") is None else f"{r['z']:+.2f}"
    verdict = "REJECTED" if r["rejected"] else "passed"
    return (f"{SCREEN_FIGHT} {verdict}: survival {r['cand_surv']:.1f} (candidate) / {r['inc_surv']:.1f} (incumbent) in 100 on {r['starts']} human-stratum "
            f"starts, difference {r['diff']:+.2f} points, SE {r['se']}, {z} SE (rejects below -{r['bound']:g})")


def run_fight(cand: Path, base_sha: str, label: str, play: Callable[[str], int], *, exam_dir: Path | None = None, loop_root: Path | None = None,
              boot: int = 400) -> dict:
    """The fight screen end to end. `play(label)` plays the candidate's missing human-stratum games and returns the exit code of
    `fight_exam.py play` (0: every game has a row). Returns `fight_reading`'s figures with `line`, or {"skipped": why, ...}."""
    import fight_exam as fe  # noqa: PLC0415
    import fight_gym as fg  # noqa: PLC0415

    exam_dir = Path(exam_dir or fe.EXAM_DIR)
    loop_root = Path(loop_root or fe.LOOP_ROOT)
    out: dict[str, Any] = {"rule": SCREEN_FIGHT, "kind": "fight", "label": label}
    human = fe.load_human_exam(exam_dir)
    if not human:
        return {**out, "skipped": "the fight exam has no human stratum"}
    inc_name = fe.gym_player_of(base_sha, exam_dir, loop_root)
    inc_path = fe.games_path(exam_dir, inc_name) if inc_name else None
    inc_games = fe.human_games(fg.read_jsonl(inc_path), human, fe.DEFAULT_CONFIG) if inc_path and inc_path.is_file() else {}
    if not inc_games:
        return {**out, "skipped": "no human-stratum games for the incumbent's fight network" + (f" (gym player {inc_name})" if inc_name else
                                                                                               " (no gym player holds these bytes)")}
    rc = play(label)
    games = fe.human_games(fg.read_jsonl(fe.games_path(exam_dir, label)), human, fe.DEFAULT_CONFIG)
    expected = len(human) * fe.DRAWS
    if rc != 0 or len(games) < expected:
        return {**out, "skipped": f"the candidate's human-stratum games are not all played (exit {rc}, {len(games)} of {expected})",
                "cand_games": len(games), "incumbent": inc_name}
    r = fight_reading(games, inc_games, human, boot=boot)
    r.update(kind="fight", label=label, incumbent=inc_name)
    if not r["comparable"]:
        r["skipped"] = f"engines differ (candidate {','.join(r['cand_engine']) or '?'}, incumbent {','.join(r['inc_engine']) or '?'}): not compared"
    r["line"] = fight_line(r)
    return r


# ---------------------------------------------------------------------------------------------------------- screen:prior


def prior_reading(cand_total: float, base_total: float, per_kind: dict | None = None, decisions: int = 0) -> dict:
    """The verdict from the two priors' held-out log-losses (lower is better): rejected when the candidate's is worse."""
    return {"rule": SCREEN_PRIOR, "kind": "prior", "cand_logloss": round(cand_total, 5), "inc_logloss": round(base_total, 5),
            "diff": round(cand_total - base_total, 5), "decisions": decisions, "per_kind": per_kind or {}, "rejected": bool(cand_total > base_total)}


def prior_line(r: dict) -> str:
    verdict = "REJECTED" if r["rejected"] else "passed"
    return (f"{SCREEN_PRIOR} {verdict}: held-out human-choice log-loss {r['cand_logloss']:.4f} (candidate) / {r['inc_logloss']:.4f} (incumbent), "
            f"difference {r['diff']:+.4f}, on {r['decisions']} decisions of the validation fifth (lower is better)")


def heldout_logloss(prior_path: Path, dec_dir: Path, kinds: list[str]) -> dict:
    """One prior's held-out human-choice log-loss: `human_prior.score_total` over the validation fifth's decisions of `kinds` (the exam fifth is
    asserted absent), with each kind's mean negative log-likelihood. CPU, numpy."""
    import human_prior as hp  # noqa: PLC0415
    import human_starts as hs  # noqa: PLC0415

    ds = hp.Dataset.load(dec_dir, [hp.VAL_FIFTH], None, kinds)
    hs.assert_no_exam(ds.run_ids, hs.read_exam_ids(hp.EXAM_IDS))
    valid = ~hp._bool_unavailable(ds, False, True, True)  # noqa: SLF001
    prior = hp.NumpyPrior(Path(prior_path))
    m = hp.metrics_from_scores(ds, hp.row_scores_np(ds, None, prior), valid)
    scored = ("card", "ancient", "rest_kind", "rest_smith_target", "shop", "route")  # the kinds `score_total` adds up
    return {"total": hp.score_total(m), "kinds": {k: m[k]["nll"] for k in scored if k in m},
            "decisions": int(sum(m[k]["n"] for k in ("card", "ancient", "rest_kind", "shop", "route") if k in m))}


def shared_kinds(cand_path: Path, base_path: Path) -> list[str]:
    """The decision kinds both priors were fitted on (their `meta["kinds"]`), in the fit's order."""
    import human_prior as hp  # noqa: PLC0415

    a, b = hp.NumpyPrior(Path(cand_path)).meta.get("kinds"), hp.NumpyPrior(Path(base_path)).meta.get("kinds")
    default = ["card", "ancient", "rest", "route"]
    return [k for k in (a or default) if k in (b or default)]


def run_prior(cand: Path, base: Path, *, dec_dir: Path | None = None, score: Callable[[Path, Path, list[str]], dict] = heldout_logloss) -> dict:
    """The prior screen end to end: both priors on the held-out human decisions. `score(prior path, decisions folder, kinds)` is the scorer."""
    import human_prior as hp  # noqa: PLC0415

    out: dict[str, Any] = {"rule": SCREEN_PRIOR, "kind": "prior"}
    dirs = [Path(dec_dir)] if dec_dir else [hp.OUT_DIR / "decisions-v1", hp.OUT_DIR / "decisions"]
    dec = next((d for d in dirs if d.is_dir() and any(d.glob("shard-*.npz"))), None)
    if dec is None:
        return {**out, "skipped": "the held-out human decisions are not here (" + ", ".join(str(d) for d in dirs) + ")"}
    cp, bp = Path(cand) / "human_prior.npz", Path(base) / "human_prior.npz"
    if not (cp.is_file() and bp.is_file()):
        return {**out, "skipped": "one of the two has no human_prior.npz"}
    kinds = shared_kinds(cp, bp)
    c, b = score(cp, dec, kinds), score(bp, dec, kinds)
    r = prior_reading(c["total"], b["total"], {k: {"cand": c["kinds"].get(k), "inc": b["kinds"].get(k)} for k in sorted(set(c["kinds"]) | set(b["kinds"]))},
                      c.get("decisions", 0))
    r["decisions_folder"] = dec.name
    r["line"] = prior_line(r)
    return r
