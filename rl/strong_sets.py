#!/usr/bin/env python3
"""Strong sets by achievement (D55 addendum 15, 2026-11-08): fit the human deck prior on the runs that achieved the most.

    python rl/strong_sets.py filter   [--export F] [--exam-ids F] [--work DIR]
    python rl/strong_sets.py fit      --name baseline|s-win|s-high|s-win+s-high [--work DIR] [--threads N]
    python rl/strong_sets.py evaluate [--work DIR] [--incumbent weights.npz]
    python rl/strong_sets.py candidate [--base gen13/candidate-stakes] [--out gen14/candidate-best-players]

The export's per-player key is blank on solo runs, so strength is read from what a run achieved. Two sets, from the Ironclad solo
standard-mode finished unmodified vanilla-act runs of the Spire Codex export (the exam fifth out of both):

* **S-win**: Ascension 10 runs that won.
* **S-high**: Ascension 11 to 20 runs, any outcome.

`filter` streams the export once and writes three working files into the work folder (never elsewhere): `a10_pool.jsonl.gz` (every
Ascension 10 run: the all-player pool, the reproduction baseline and the parent of S-win), `s_win.jsonl.gz`, `s_high.jsonl.gz`, and
`filter_stats.json` (counts only). A written run drops `player_token`, `badges` and the player's id. The decision shards are built with
`human_prior.py build --corpus <file> --ascensions 10|11-20` (the same code path as the prior in generation 13); `fit` is `human_prior.fit_one`
(the generation 11 start, winners 1x, fifths 1 to 3 train, fifth 4 validates) on one set's decisions, on the CPU; `evaluate` scores every
fit on the held-out fifth (fifth 4, trained on by no fit) of the won A10 runs (a) and of all A10 runs (b), plus a ranking measure (c), and
applies the selection rule (`select_fit`). Aggregates only: no run is named and no cell under 20 runs is shown.
"""

from __future__ import annotations

import argparse
import gzip
import json
import os
import re
import sys
import time
import types
from collections import Counter
from collections.abc import Iterable
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import human_prior as hp  # noqa: E402
import human_starts as hs  # noqa: E402

DATA = Path("D:/spireline/data/spire-codex")
EXPORT = DATA / "2026-11-06" / "runs-v0.107.1.jsonl.gz"
EXAM_IDS = DATA / "human-exam-fifth-ids-2026-11-06.txt"
WORK = DATA / "work-2026-11-08"
BUILD_ID = "v0.107.1"
SPIRE_CODEX = hp.SPIRE_CODEX
A10 = (10,)
HIGH = tuple(range(11, 21))
MIN_CELL = hp.MIN_CELL
TIE_NATS = 0.002  # two fits whose held-out log-loss on the won runs differ by less than this are tied
LIMIT_B_NATS = 0.02  # a strong fit may not be worse than the all-player fit on the all-player runs by more than this
FILES = {"a10": "a10_pool.jsonl.gz", "s-win": "s_win.jsonl.gz", "s-high": "s_high.jsonl.gz"}
# the fits: name -> the (shard folder, won only) parts of its decisions; "prior-a10" holds the Ascension 10 pool, "prior-high" the S-high runs (each
# folder is a `human_prior.py build --out-dir` with its donors.npz beside the `decisions` shards)
FITS = {
    "baseline": [("prior-a10/decisions", False)],
    "s-win": [("prior-a10/decisions", True)],
    "s-high": [("prior-high/decisions", False)],
    "s-win+s-high": [("prior-a10/decisions", True), ("prior-high/decisions", False)],
}
# an informational arm, scored beside the four but outside the selection rule: S-win's decisions, warm-started from the baseline's weights
ARMS = {**FITS, "s-win-ft": [("prior-a10/decisions", True)]}
KINDS = ("card", "ancient", "rest")
ASC_RX = re.compile(r'"ascension":(\d+),')


# ------------------------------------------------------------------------------------------------------------------ the filter


def run_key(r: dict) -> str:
    """The run's identity for deduplication and the fifths: `human_starts.run_id` (the run hash, else seed and start time)."""
    return hs.run_id(r)


def sets_of(r: dict) -> tuple[str, ...]:
    """The strong-set names of a run that already passed `reject_reason`: 'a10' for every Ascension 10 run, plus 's-win' for a win;
    's-high' for Ascension 11 to 20 (any outcome)."""
    asc = r.get("ascension")
    if asc in A10:
        return ("a10", "s-win") if r.get("win") else ("a10",)
    if asc in HIGH:
        return ("s-high",)
    return ()


def reject_reason(r: dict, exam: frozenset, seen: set) -> str | None:
    """Why an Ironclad run is not in any set, or None. The rules are the 11-06 filter's (standard mode, not abandoned, one player, no
    modifiers, build v0.107.1, one line per run hash) with the fork's own (vanilla acts) and the exam fifth out, for the levels 10 to 20."""
    if r.get("ascension") not in A10 + HIGH:
        return "ascension"
    if not hs.human_ok(r, A10 + HIGH):
        pl = r.get("players") or []
        if r.get("game_mode") != "standard":
            return "game_mode"
        if r.get("was_abandoned"):
            return "abandoned"
        if len(pl) != 1:
            return "multiplayer"
        if r.get("modifiers"):
            return "modifiers"
        return "character"
    if r.get("build_id") != BUILD_ID:
        return "build"
    key = run_key(r)
    if key in seen:
        return "duplicate"
    if hs.is_exam(key, exam):
        seen.add(key)
        return "exam_fifth"
    if not hs.vanilla_acts(r):
        return "modded_acts"
    seen.add(key)
    return None


def scrub(r: dict) -> dict:
    """The run as it is written to a working file: no player token, no badges, no player id (nothing about a player)."""
    r = dict(r)
    r.pop("player_token", None)
    pls = []
    for p in r.get("players") or []:
        p = dict(p)
        p.pop("badges", None)
        if "id" in p:
            del p["id"]
        pls.append(p)
    r["players"] = pls
    return r


def acts_reached(r: dict) -> int:
    """The last act the run entered: how many of the record's per-act map point lists hold a point."""
    return sum(1 for a in (r.get("map_point_history") or []) if a)


def filter_stream(lines: Iterable[str], exam: frozenset, outs: dict[str, object] | None = None) -> dict:
    """Stream raw export lines; write each kept run (scrubbed) to the file of each set it belongs to (`outs`: set -> text file), and return
    the counts: lines read, Ironclad lines, rejections by reason, and per set the runs, wins, runs by fifth and runs by acts reached."""
    stats: Counter = Counter()
    per_set: dict[str, Counter] = {s: Counter() for s in FILES}
    seen: set = set()
    for line in lines:
        stats["lines"] += 1
        if '"character":"CHARACTER.IRONCLAD"' not in line:
            continue
        m = ASC_RX.search(line)
        if m is not None and int(m.group(1)) not in A10 + HIGH:
            continue  # the cheap screen: most Ironclad lines are other levels, and parsing a run is the pass's cost
        try:
            r = json.loads(line)
        except ValueError:
            stats["unparsable"] += 1
            continue
        pl = r.get("players") or []
        if len(pl) == 1 and pl[0].get("character") != "CHARACTER.IRONCLAD":
            continue
        asc = r.get("ascension")
        if asc not in A10 + HIGH:
            continue
        stats["ironclad_asc_10_to_20"] += 1
        why = reject_reason(r, exam, seen)
        if why:
            stats[f"rejected|{why}"] += 1
            continue
        names = sets_of(r)
        text = json.dumps(scrub(r), separators=(",", ":")) + "\n"
        fifth = hs.exam_bucket(run_key(r))
        act_n = acts_reached(r)
        for name in names:
            if outs is not None:
                outs[name].write(text)
            c = per_set[name]
            c["runs"] += 1
            c["wins"] += bool(r.get("win"))
            c[f"fifth{fifth}"] += 1
            c[f"last_act{max(1, min(3, act_n))}"] += 1
    return {"read": dict(stats), "sets": {s: dict(c) for s, c in per_set.items()}}


def cmd_filter(a: argparse.Namespace) -> int:
    hs.below_normal()
    work = Path(a.work)
    work.mkdir(parents=True, exist_ok=True)
    exam = hs.read_exam_ids(a.exam_ids)
    t0 = time.time()
    outs = {}
    paths = {s: work / f for s, f in FILES.items()}
    try:
        for s, p in paths.items():
            outs[s] = gzip.open(Path(str(p) + ".tmp"), "wt", encoding="utf-8")
        with gzip.open(a.export, "rt", encoding="utf-8") as fh:
            doc = filter_stream(fh, exam, outs)
    finally:
        for f in outs.values():
            f.close()
    for p in paths.values():
        os.replace(Path(str(p) + ".tmp"), p)
    doc.update(written_by="rl/strong_sets.py filter (counts only)", source=SPIRE_CODEX, export=Path(a.export).name,
               exam_fifth="the ids file united with hash bucket 0 (human_starts.is_exam)", seconds=round(time.time() - t0, 1),
               files={s: p.name for s, p in paths.items()})
    (work / "filter_stats.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(json.dumps(doc, indent=1))
    return 0


# ---------------------------------------------------------------------------------------------------------------- the datasets


def load_parts(work: Path, name: str, fifths: Iterable[int], donors: dict | None, kinds: Iterable[str] = KINDS):
    """The decisions of fit `name` in `fifths`: the parts of `FITS[name]` joined (a won-only part keeps the winning runs' decisions)."""
    parts = [(Path(work) / folder, won_only) for folder, won_only in ARMS[name]]
    return hp.Dataset.load(parts, fifths, donors, kinds=list(kinds))


def heldout_sets(work: Path, donors: dict | None):
    """The held-out fifth (VAL_FIFTH, trained on by no fit) of the all-player A10 pool; (a) is its won runs (the decisions flagged `won`)."""
    return hp.Dataset.load(Path(work) / "prior-a10" / "decisions", [hp.VAL_FIFTH], donors, kinds=list(KINDS))


def runs_of(ds: "hp.Dataset") -> set[str]:
    """The runs a loaded dataset's decisions come from (its `run_ids` list every run of the shards read, whatever fifth they are in)."""
    return {ds.run_ids[i] for i in np.unique(ds.dec[:, hp.C["run"]])}


def no_overlap(train_run_ids: Iterable[str], held_run_ids: Iterable[str]) -> int:
    """Raise when a held-out run is among a fit's training runs (it never should be: the fifths are disjoint); returns the held-out count."""
    held = set(held_run_ids)
    clash = held & set(train_run_ids)
    if clash:
        raise AssertionError(f"strong_sets: {len(clash)} held-out runs are in a training set")
    return len(held)


# ---------------------------------------------------------------------------------------------------------------------- the fit


def cmd_fit(a: argparse.Namespace) -> int:
    os.environ["CUDA_VISIBLE_DEVICES"] = "-1"  # the GPU belongs to the lab's judgement: a fit here is CPU only
    import torch  # noqa: PLC0415

    if torch.cuda.is_available():
        raise SystemExit("strong_sets fit: a GPU is visible; this fit is CPU only")
    torch.set_num_threads(max(1, a.threads))
    hs.below_normal()
    t0 = time.time()
    work = Path(a.work)
    donors = hp.load_donors(work / "prior-a10" / "donors.npz")
    ds_tr = load_parts(work, a.name, hp.TRAIN_FIFTHS, donors)
    ds_va = load_parts(work, a.name, [hp.VAL_FIFTH], donors)
    exam = hs.read_exam_ids(a.exam_ids)
    hs.assert_no_exam(ds_tr.run_ids, exam)
    hs.assert_no_exam(ds_va.run_ids, exam)
    no_overlap(runs_of(ds_tr), runs_of(ds_va))
    kinds = list(KINDS)
    runs_tr, runs_va = len(runs_of(ds_tr)), len(runs_of(ds_va))
    print(f"strong_sets fit {a.name}: train {len(ds_tr.dec):,} decisions of {runs_tr:,} runs; validation {len(ds_va.dec):,} decisions "
          f"of {runs_va:,} runs ({time.time() - t0:.0f} s)", flush=True)
    dev = torch.device("cpu")
    gt, gv = hp.Gpu(ds_tr, donors, dev, 1.0), hp.Gpu(ds_va, donors, dev, 1.0)
    args = types.SimpleNamespace(seed=a.seed, epochs=a.epochs, patience=a.patience, batch=a.batch, with_shops=False)
    init = f"prior:{a.init_from}" if a.init_from else "gen11"
    res = hp.fit_one(gt, gv, init, 1.0, args, kinds, Path(a.judge))
    out = Path(a.work) / "fits" / a.name
    out.mkdir(parents=True, exist_ok=True)
    meta = {"kinds": kinds, "feature_version": hp.stack().F.FEATURE_VERSION, "map": "donor", "source": SPIRE_CODEX, "variant": "gen11-w1" if not a.init_from else "warm-w1",
            "strong_set": a.name, "judge": str(a.judge), "inputs": "judge dense 350 + counts 1513; one logit per option"}
    hp.export_net(res["net"], out / "weights.npz", meta)
    valid_tr = ~hp._bool_unavailable(ds_tr, False, True, True)
    fit_doc = {"written_by": "rl/strong_sets.py fit (aggregates only)", "source": SPIRE_CODEX, "set": a.name, "parts": ARMS[a.name],
               "train_fifths": list(hp.TRAIN_FIFTHS), "validation_fifth": hp.VAL_FIFTH, "device": "cpu", "threads": a.threads, "seed": a.seed,
               "init": "the generation 11 judgement's trunk" if not a.init_from else f"warm start from the fitted prior {hp_sha(Path(a.init_from))}",
               "kinds_fitted": kinds, "runs": {"train": runs_tr, "validation": runs_va},
               "trained_on_decisions": {k: int(valid_tr[ds_tr.kind_idx[k]].sum()) for k in kinds},
               "best_epoch": res["best"]["epoch"], "epochs_run": res["epochs_run"], "val_score": res["best"]["score"],
               "val_metrics": res["best"]["metrics"], "history": res["history"], "seconds": res["seconds"],
               "wall_seconds": round(time.time() - t0, 1), "torch": torch.__version__}
    (out / "fit.json").write_text(json.dumps(fit_doc, indent=1), encoding="utf-8")
    print(json.dumps({"set": a.name, "best_epoch": fit_doc["best_epoch"], "val_score": fit_doc["val_score"], "wall_seconds": fit_doc["wall_seconds"]}))
    return 0


# ----------------------------------------------------------------------------------------------------------------- the measures


def _top_rank(s: np.ndarray, chosen: np.ndarray) -> np.ndarray:
    """1 + the number of options scored above the chosen one (1 = the prior's first choice)."""
    return 1 + (s > s[np.arange(len(chosen)), chosen][:, None]).sum(1)


def per_decision(ds: "hp.Dataset", scores: np.ndarray, valid: np.ndarray) -> dict[str, dict[str, np.ndarray]]:
    """Per decision, for the kinds the fit trains: the human's negative log-likelihood, top-1 hit, rank and the run it came from (a
    run index into `ds.run_ids`). Same definitions as `human_prior.metrics_from_scores` (a test pins the means equal). Keys: card,
    ancient, rest_kind, rest_smith_target. Arrays: run, act, nll, hit, rank (card, ancient, smith target), take (card: the human took
    a card), split (card: half of a merged screen)."""
    C, dec = hp.C, ds.dec
    out: dict[str, dict[str, np.ndarray]] = {}
    for kind in ("card", "ancient", "rest"):
        idx = ds.kind_idx[kind]
        idx = idx[valid[idx]]
        if not len(idx):
            continue
        m = ds.matrix(idx)
        ok = m >= 0
        s = np.where(ok, scores[np.maximum(m, 0)], -np.inf)
        act = dec[idx, C["act"]]
        run = dec[idx, C["run"]]
        chosen = dec[idx, C["chosen"]]
        lse = hp._lse(s)
        top = s.argmax(1)
        if kind in ("card", "ancient"):
            nll = lse - s[np.arange(len(idx)), chosen]
            rec = {"run": run, "act": act, "nll": nll, "hit": (top == chosen).astype(np.float64), "rank": _top_rank(s, chosen).astype(np.float64)}
            if kind == "card":
                rec["take"] = (dec[idx, C["aux"]] == 0).astype(np.float64)
                rec["split"] = ((dec[idx, C["flags"]] & hp.F_SPLIT) != 0).astype(np.float64)
            out[kind] = rec
            continue
        role = np.where(ok, ds.role[np.maximum(m, 0)], -1)
        aux = dec[idx, C["aux"]]
        sk = np.where(role == aux[:, None], s, -np.inf)
        top_role = np.take_along_axis(role, top[:, None], 1)[:, 0]
        out["rest_kind"] = {"run": run, "act": act, "nll": lse - hp._lse(sk), "hit": (top_role == aux).astype(np.float64)}
        smith = (aux == hp.ROLE_SMITH) & (chosen >= 0)
        if smith.any():
            ss = np.where(role == hp.ROLE_SMITH, s, -np.inf)[smith]
            ch = chosen[smith]
            rank = _top_rank(ss, ch).astype(np.float64)
            out["rest_smith_target"] = {"run": run[smith], "act": act[smith], "nll": hp._lse(ss) - ss[np.arange(len(ch)), ch],
                                        "hit": (ss.argmax(1) == ch).astype(np.float64), "rank": rank, "top3": (rank <= 3).astype(np.float64)}
    return out


POOLED = ("card", "ancient", "rest_kind")  # the decisions the headline log-loss and top-1 pool (one per decision, one softmax each)


def pooled(pd: dict[str, dict[str, np.ndarray]], mask_run: np.ndarray | None = None) -> dict[str, np.ndarray]:
    """The pooled arrays of `POOLED` kinds (run, nll, hit); `mask_run` (bool per run index) keeps only the decisions of those runs."""
    runs, nll, hit = [], [], []
    for k in POOLED:
        if k not in pd:
            continue
        keep = np.ones(len(pd[k]["run"]), dtype=bool) if mask_run is None else mask_run[pd[k]["run"]]
        runs.append(pd[k]["run"][keep])
        nll.append(pd[k]["nll"][keep])
        hit.append(pd[k]["hit"][keep])
    return {"run": np.concatenate(runs), "nll": np.concatenate(nll), "hit": np.concatenate(hit)}


def summary(pd: dict[str, dict[str, np.ndarray]], mask_run: np.ndarray | None = None) -> dict:
    """The held-out figures of one fit on one set of runs: the pooled log-loss and top-1, each kind's, and the ranking measure (c):
    the mean rank of the human's card among the card-reward options and the share of smith targets inside the prior's top three."""
    p = pooled(pd, mask_run)
    out: dict = {"decisions": int(len(p["nll"])), "log_loss": float(p["nll"].mean()), "top1": float(p["hit"].mean()), "kinds": {}}
    for k, rec in pd.items():
        keep = np.ones(len(rec["run"]), dtype=bool) if mask_run is None else mask_run[rec["run"]]
        if not keep.any():
            continue
        e = {"n": int(keep.sum()), "log_loss": float(rec["nll"][keep].mean()), "top1": float(rec["hit"][keep].mean())}
        if "rank" in rec:
            e["mean_rank"] = float(rec["rank"][keep].mean())
        if "top3" in rec:
            e["top3"] = float(rec["top3"][keep].mean())
        out["kinds"][k] = e
    c = out["kinds"]
    out["ranking"] = {"card_mean_rank": c.get("card", {}).get("mean_rank"), "smith_target_top3": c.get("rest_smith_target", {}).get("top3"),
                      "smith_target_n": c.get("rest_smith_target", {}).get("n")}
    return out


def cells(pd: dict[str, dict[str, np.ndarray]], mask_run: np.ndarray | None = None) -> dict[str, dict]:
    """The published cells (kind x act, with 'all'): decisions and distinct runs in each. A cell under MIN_CELL runs is not shown."""
    out: dict[str, dict] = {}
    for k, rec in pd.items():
        keep = np.ones(len(rec["run"]), dtype=bool) if mask_run is None else mask_run[rec["run"]]
        for label, sel in (("all", keep), ("act1", keep & (rec["act"] == 0)), ("act2", keep & (rec["act"] == 1)), ("act3", keep & (rec["act"] == 2))):
            out[f"{k}|{label}"] = {"decisions": int(sel.sum()), "runs": int(len(np.unique(rec["run"][sel])))}
    return out


def paired_bootstrap(diff_run: np.ndarray, diff: np.ndarray, n_runs: int, draws: int = 2000, seed: int = 20261108) -> dict:
    """The mean per-decision difference `diff` (fit minus baseline, nats) with a 95% interval from resampling whole runs. `diff_run` is the
    run index of each decision; runs are the unit (the decisions of a run share its player)."""
    uniq, inv = np.unique(diff_run, return_inverse=True)
    s = np.bincount(inv, weights=diff, minlength=len(uniq))
    n = np.bincount(inv, minlength=len(uniq)).astype(np.float64)
    rng = np.random.default_rng(seed)
    pick = rng.integers(0, len(uniq), size=(draws, len(uniq)))
    means = s[pick].sum(1) / np.maximum(n[pick].sum(1), 1)
    return {"mean": float(diff.mean()), "lo95": float(np.percentile(means, 2.5)), "hi95": float(np.percentile(means, 97.5)),
            "runs": int(len(uniq)), "draws": draws}


def select_fit(rows: dict[str, dict], baseline: str = "baseline", tie: float = TIE_NATS, limit_b: float = LIMIT_B_NATS) -> dict:
    """The selection rule. `rows[name]` = {a, b, size}: the held-out log-loss on the won runs (a) and on all A10 runs (b), and the fit's
    training size (runs). A strong fit is eligible when its (b) is not worse than the baseline's by more than `limit_b`. The best (a)
    among the baseline and the eligible fits wins; fits within `tie` of it tie, and a tie goes to the larger set. When the baseline is in
    the tied group no strong fit beats it and nothing is selected."""
    base = rows[baseline]
    strong = [n for n in rows if n != baseline]
    eligible = [n for n in strong if rows[n]["b"] - base["b"] <= limit_b + 1e-12]
    out: dict = {"selected": None, "eligible": eligible, "refused_on_b": [n for n in strong if n not in eligible],
                 "rule": f"best held-out log-loss on the won runs (a) among the baseline and the fits whose all-run log-loss (b) is not worse than the "
                         f"baseline's by more than {limit_b} nats; fits within {tie} nats of the best tie and the larger set wins; none beats the baseline "
                         f"when the baseline is among the tied"}
    pool = [baseline] + eligible
    best = min(rows[n]["a"] for n in pool)
    tied = [n for n in pool if rows[n]["a"] <= best + tie + 1e-12]
    out["tied"] = tied
    if baseline in tied:
        out["why"] = "no strong fit beats the all-player baseline on the won runs (a) by more than the tie margin" if eligible else \
            "no strong fit passes the all-run limit (b), and none beats the baseline on (a)"
        return out
    pick = max(tied, key=lambda n: (rows[n]["size"], -rows[n]["a"]))
    out["selected"] = pick
    out["why"] = f"{pick} has the best held-out log-loss on the won runs ({rows[pick]['a']:.4f} against the baseline's {base['a']:.4f}) and passes the limit on (b)"
    return out


def cmd_evaluate(a: argparse.Namespace) -> int:
    hs.below_normal()
    t0 = time.time()
    work = Path(a.work)
    donors = hp.load_donors(work / "prior-a10" / "donors.npz")
    ds = heldout_sets(work, donors)
    exam = hs.read_exam_ids(a.exam_ids)
    hs.assert_no_exam(ds.run_ids, exam)
    valid = ~hp._bool_unavailable(ds, False, True, True)
    won_dec = ds.dec[:, hp.C["won"]] == 1
    n_runs = len(ds.run_ids)
    run_won = np.zeros(n_runs, dtype=bool)
    run_won[ds.dec[won_dec, hp.C["run"]]] = True
    models = {n: work / "fits" / n / "weights.npz" for n in ARMS}
    if a.incumbent:
        models["incumbent (2ebfe6d7)"] = Path(a.incumbent)
    per: dict[str, dict] = {}
    table: dict[str, dict] = {}
    for name, path in models.items():
        if not path.is_file():
            print(f"strong_sets evaluate: no weights for {name} ({path.name}); skipped", file=sys.stderr)
            continue
        prior = hp.NumpyPrior(path)
        scores = hp.row_scores_np(ds, donors, prior)
        pd = per_decision(ds, scores, valid)
        per[name] = pd
        table[name] = {"a_won_runs": summary(pd, run_won), "b_all_runs": summary(pd, None), "sha256_8": hp_sha(path)}
    base = per.get("baseline")
    for name, pd in per.items():
        if name == "baseline" or base is None:
            continue
        pa, pb = pooled(pd, run_won), pooled(base, run_won)
        table[name]["a_minus_baseline"] = paired_bootstrap(pa["run"], pa["nll"] - pb["nll"], n_runs)
        qa, qb = pooled(pd, None), pooled(base, None)
        table[name]["b_minus_baseline"] = paired_bootstrap(qa["run"], qa["nll"] - qb["nll"], n_runs)
    fits = [n for n in FITS if n in table]  # the selection rule reads the four fits; an informational arm is scored beside them
    sizes = {}
    for n in fits:
        fj = work / "fits" / n / "fit.json"
        sizes[n] = json.loads(fj.read_text(encoding="utf-8"))["runs"]["train"] if fj.is_file() else 0
    rows = {n: {"a": table[n]["a_won_runs"]["log_loss"], "b": table[n]["b_all_runs"]["log_loss"], "size": sizes[n]} for n in fits}
    sel = select_fit(rows) if "baseline" in rows else {"selected": None, "why": "no baseline fit"}
    held = {"runs_all": int(len(runs_of(ds))), "runs_won": int(run_won.sum()),
            "cells_a": cells(per["baseline"], run_won) if "baseline" in per else {},
            "cells_b": cells(per["baseline"], None) if "baseline" in per else {}}
    for k in ("cells_a", "cells_b"):
        held[k + "_under_min"] = sum(1 for c in held[k].values() if c["runs"] < MIN_CELL or c["decisions"] < MIN_CELL)
    doc = {"written_by": "rl/strong_sets.py evaluate (aggregates only)", "source": SPIRE_CODEX, "heldout_fifth": hp.VAL_FIFTH, "heldout": held,
           "fits": table, "training_runs": sizes, "selection": sel, "seconds": round(time.time() - t0, 1)}
    (work / "heldout.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    print(json.dumps({"selection": sel, "rows": rows}, indent=1))
    return 0


def hp_sha(path: Path) -> str:
    return file_sha256(path)[:8]


def file_sha256(path: Path) -> str:
    import hashlib  # noqa: PLC0415

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


# --------------------------------------------------------------------------------------------------------------- the candidate

NOT_COPIED = ("arms", "promotion.json")  # a candidate's arms and verdict belong to the candidate that played them
EDITED = ("agent.json", "train.json", "model.json", "human_prior.npz")
GATE_NOTE = "no marker, no promotion.json, no arms: the orchestrator writes the marker once the lab has judged the earlier candidates"


def _swap(obj, pairs: list[tuple[str, str]]):
    """Every string in a JSON value with each (old, new) pair replaced."""
    if isinstance(obj, str):
        for old, new in pairs:
            obj = obj.replace(old, new)
        return obj
    if isinstance(obj, list):
        return [_swap(x, pairs) for x in obj]
    if isinstance(obj, dict):
        return {k: _swap(v, pairs) for k, v in obj.items()}
    return obj


def build_candidate(base: Path, out: Path, weights: Path, info: dict, generation: int = 14, tag: str = "best-players", now: str | None = None) -> dict:
    """`out` = a copy of the candidate `base` with only the human prior replaced by `weights`: human_prior.npz, agent.json (the macro and
    recipe carry the new prior's hash, the chooser's sha256 and source), train.json (the same, and `strong_set` with the set and its
    counts), model.json (the iteration). Every other weight file is a byte-for-byte copy; no arms, no promotion.json, no marker.
    `info` = {"set", "source", "why", "counts": {...}}. Returns the new agent.json."""
    import shutil  # noqa: PLC0415

    base, out, weights = Path(base), Path(out), Path(weights)
    if out.exists():
        raise FileExistsError(f"{out} exists: a candidate folder is written once")
    agent = json.loads((base / "agent.json").read_text(encoding="utf-8"))
    train = json.loads((base / "train.json").read_text(encoding="utf-8"))
    old_full = agent["human_prior"]["sha256"]
    new_full = file_sha256(weights)
    if new_full == old_full:
        raise ValueError("the new prior is the base's prior: nothing to replace")
    pairs = [(old_full, new_full), (old_full[:8], new_full[:8])]
    out.mkdir(parents=True)
    for f in sorted(base.iterdir()):
        if f.is_file() and f.name not in NOT_COPIED and f.name not in EDITED:
            shutil.copy2(f, out / f.name)
    shutil.copy2(weights, out / "human_prior.npz")
    now = now or time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    why = info["why"]
    agent = _swap(agent, pairs)
    agent["generation"] = generation
    agent["written"] = now
    agent["lineage"]["skips"] = sorted(set(agent["lineage"].get("skips", [])) | {12, 13})
    agent["lineage"]["decision"] = why
    agent["lineage"]["difference"] = (f"generation 13's recipe (candidate-stakes) with only the human prior replaced: pi_h fitted on {info['set']} "
                                      f"({new_full[:8]}) in place of the all-player fit ({old_full[:8]}); {GATE_NOTE}")
    agent["human_prior"]["source"] = info["source"]
    agent["built"]["why"] = why
    agent["built"]["tag"] = tag
    agent["note"] = (f"built, not trained: generation 13's recipe with the human prior fitted on {info['set']} ({new_full[:8]}); "
                     f"its E-block judges the recipe as a whole")
    train = _swap(train, pairs)
    train["candidate_generation"] = generation
    train["built"]["why"] = why
    train["built"]["written"] = now
    train["built"]["tag"] = tag
    train["built"]["chooser"]["source"] = info["source"]
    train["lineage"]["skips"] = agent["lineage"]["skips"]
    train["lineage"]["decision"] = why
    train["lineage"]["difference"] = agent["lineage"]["difference"]
    train["strong_set"] = {"set": info["set"], "source": SPIRE_CODEX, "counts": info["counts"], "prior_sha256": new_full,
                           "replaces_sha256": old_full,
                           "fit": "rl/strong_sets.py fit (the generation 11 start, winners 1x, fifths 1 to 3 train, fifth 4 validates; CPU)",
                           "note": GATE_NOTE}
    model = json.loads((base / "model.json").read_text(encoding="utf-8"))
    model["iteration"] = generation
    for name, doc in (("agent.json", agent), ("train.json", train), ("model.json", model)):
        (out / name).write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    # the checks: the macro, the recipe and the chooser name the new prior, and no other weight file moved
    assert agent["human_prior"]["sha256"] == file_sha256(out / "human_prior.npz") == new_full
    assert new_full[:8] in agent["macro"] and new_full[:8] in agent["recipe"]
    assert old_full[:8] not in agent["macro"] + agent["recipe"] + json.dumps(agent["human_prior"]) + json.dumps(train["built"]["chooser"])
    for f in base.iterdir():
        if f.is_file() and f.name not in NOT_COPIED and f.name not in EDITED:
            assert file_sha256(f) == file_sha256(out / f.name), f.name
    return agent


def cmd_candidate(a: argparse.Namespace) -> int:
    doc = json.loads((Path(a.work) / "heldout.json").read_text(encoding="utf-8"))
    sel = doc["selection"]["selected"]
    if sel is None:
        print("strong_sets candidate: the selection rule picked no fit; no candidate is built", file=sys.stderr)
        return 3
    fit = json.loads((Path(a.work) / "fits" / sel / "fit.json").read_text(encoding="utf-8"))
    counts = {"runs_train": fit["runs"]["train"], "runs_validation": fit["runs"]["validation"], "decisions_trained_on": fit["trained_on_decisions"],
              "best_epoch": fit["best_epoch"], "heldout_fifth": hp.VAL_FIFTH}
    why = ("D55 addendum 15 (owner 2026-11-08): the human prior fitted on the strong runs by achievement (" + sel + ") in place of the all-player "
           "fit, every other part of generation 13's recipe unchanged; selected by the held-out rule of docs/rl/HUMAN_PRIOR.md, section Strong sets by achievement")
    agent = build_candidate(Path(a.base), Path(a.out), Path(a.work) / "fits" / sel / "weights.npz",
                            {"set": sel, "source": f"strong-sets/{sel}/weights.npz", "why": why, "counts": counts})
    print(json.dumps({"candidate": Path(a.out).name, "macro": agent["macro"], "recipe": agent["recipe"], "set": sel}))
    return 0


# ------------------------------------------------------------------------------------------------------------------------- main


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("filter")
    f.add_argument("--export", default=str(EXPORT))
    f.add_argument("--exam-ids", default=str(EXAM_IDS))
    f.add_argument("--work", default=str(WORK))
    t = sub.add_parser("fit")
    t.add_argument("--name", required=True, choices=sorted(ARMS))
    t.add_argument("--work", default=str(WORK))
    t.add_argument("--exam-ids", default=str(EXAM_IDS))
    t.add_argument("--judge", default=str(hp.GEN11))
    t.add_argument("--threads", type=int, default=1)
    t.add_argument("--epochs", type=int, default=30)
    t.add_argument("--patience", type=int, default=3)
    t.add_argument("--batch", type=int, default=256)
    t.add_argument("--seed", type=int, default=20261105)
    t.add_argument("--init-from", default="", help="a warm start from a fitted prior's weights.npz (an informational arm; the four fits start from the generation 11 trunk)")
    e = sub.add_parser("evaluate")
    e.add_argument("--work", default=str(WORK))
    e.add_argument("--exam-ids", default=str(EXAM_IDS))
    e.add_argument("--incumbent", default="")
    c = sub.add_parser("candidate")
    c.add_argument("--work", default=str(WORK))
    c.add_argument("--base", default="D:/spireline/referee/agent/gen13/candidate-stakes")
    c.add_argument("--out", default="D:/spireline/referee/agent/gen14/candidate-best-players")
    a = ap.parse_args(argv)
    return {"filter": cmd_filter, "fit": cmd_fit, "evaluate": cmd_evaluate, "candidate": cmd_candidate}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
