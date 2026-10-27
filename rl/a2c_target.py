"""Round A2c: which policy target the combat trainer should use -- root visits or the completed-Q softmax.

    python rl/a2c_target.py train   --out WORK [--gen 3 --temps 3,10,30]      # GPU venv: one window, one net per arm
    python rl/a2c_target.py heldout --out WORK                                # GPU venv: choose the temperature
    python rl/a2c_target.py devset  --out WORK [--procs 8 --reps 12]          # emulator: net alone and the search
    python rl/a2c_target.py report  --out WORK

`train` builds the window `gen0.py train --gen N` would (gen_loop.window_manifests, build_training_set with seed N)
once and fits one prior per arm with gen_loop.train_combat_prior: same data, epochs and seed. `heldout` scores each
net on train_combat's held-out runs by the held-out searches' own values: the expected completed-Q regret of the
net's policy (max over legal actions of completedQ minus its policy-weighted mean), which favours neither target;
the temperature is the completed-Q arm with the lowest. `devset` plays the A3 dev set (positions labelled by long
resampled searches, rl/a3_devset.py): (1) each net alone, its prior over root moves exactly as the search splits it
(identical cards pooled, an action's share split evenly among its targets), top-1 agreement by action and regret
in HP; (2) the search at a3-v1 settings with each prior, the same seeds for every arm, regret of its pick.
`report` gives population-weighted means and paired bootstrap intervals (a3_devset's weights and paired_boot).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import gen0  # noqa: E402
import gen_loop  # noqa: E402

DEVSET = gen0.ROOT.parents[1] / "referee" / "agent" / "devset"
RUNS = gen0.ROOT.parents[1] / "referee" / "lab" / "runs"
BASE = "arm-visits"


def arm_dirs(out: Path) -> dict[str, Path]:
    return {p.name: p for p in sorted(Path(out).glob("arm-*")) if (p / "combat_policy.pt").is_file()}


def train(a) -> None:
    out = Path(a.out)
    t0 = time.perf_counter()
    old = json.loads((out / "train.json").read_text()) if (out / "train.json").is_file() else None
    if a.data:  # an earlier `train`'s training set: the lab's window grows, and 6,400 fights drawn from a larger
        with np.load(a.data) as b:  # window are a different sample (and a different split by run)
            data = {k: b[k] for k in b.files}
        info = old or {"gen": a.gen, "arms": {}}
    else:
        data = gen_loop.build_training_set(gen_loop.window_manifests(Path(a.runs), a.gen), seed=a.gen)
        info = {"gen": a.gen, "build_s": round(time.perf_counter() - t0, 1), "samples": int(data["lengths"].size),
                "fights_available": int(data["_fights_available"][0]), "fights_kept": int(data["_fights_kept"][0]),
                "samples_by_generation": {str(int(g)): int(n) for g, n in data["_samples_by_generation"]},
                "arms": old["arms"] if old else {}}
    arms = [] if a.temps_only else [(BASE, "visits", None)]
    arms += [(f"arm-cq{t:g}", "completed_q", t) for t in a.temps]
    for name, kind, temp in arms:
        d = out / name
        doc = gen_loop.train_combat_prior(data, d, device=a.device or None, epochs=a.epochs, seed=a.gen,
                                          policy_target=kind, q_temperature=temp)
        gen_loop.write_prior_model_json(d, a.gen + 1)
        info["arms"][name] = {k: doc[k] for k in ("policy_target", "q_temperature", "wall_min", "report")}
        if name != BASE:  # one copy of the training set is enough (heldout reads the visit arm's)
            (d / "work" / "train.npz").unlink(missing_ok=True)
        gen_loop.write_json(out / "train.json", info)
        print(name, json.dumps(info["arms"][name]), flush=True)


def heldout(a) -> None:
    import torch  # noqa: PLC0415
    from combat_net import CombatNet, collate  # noqa: PLC0415
    from combat_tokens import F_ENDTURN_DEATH, F_ENDTURN_KNOWN, F_ENDTURN_MEAN, F_ENDTURN_WORST, VOCAB, Tokens  # noqa: PLC0415

    out = Path(a.out)
    with np.load(out / BASE / "work" / "train.npz") as b:
        blob = {k: b[k] for k in ("types", "ids", "owners", "feats", "lengths", "fights", "visits", "legal", "qvals")}
    rng = np.random.default_rng(0)  # train_combat.train's split by `fights` (the run), the same lines
    uniq = np.unique(blob["fights"])
    shuffled = uniq[rng.permutation(len(uniq))]
    val = np.flatnonzero(np.isin(blob["fights"], shuffled[int(0.85 * len(shuffled)):]))
    off = np.concatenate([[0], np.cumsum(blob["lengths"])])
    states = [Tokens(*(blob[k][off[i]:off[i + 1]] for k in ("types", "ids", "owners", "feats"))) for i in val]
    legal, visits, q = blob["legal"][val] > 0, blob["visits"][val], blob["qvals"][val].astype(np.float64)
    keep = np.where(legal, visits, 0).sum(1) > 0
    seen = legal & (visits > 0) & np.isfinite(q)
    w = np.where(seen, visits, 0.0)
    root = (w * np.where(seen, q, 0)).sum(1, keepdims=True) / np.maximum(w.sum(1, keepdims=True), 1e-9)
    cq = np.where(legal, np.where(seen, q, root), -np.inf)
    best = cq.max(1)
    dev = torch.device(a.device or ("cuda" if torch.cuda.is_available() else "cpu"))
    res = {"rows": int(keep.sum()), "runs": int(len(shuffled) - int(0.85 * len(shuffled))), "arms": {}}
    for name, d in arm_dirs(out).items():
        ck = torch.load(d / "combat_policy.pt", map_location=dev, weights_only=True)
        model = CombatNet(VOCAB, ck["n_buckets"], d_model=ck["d_model"], layers=ck["layers"], heads=ck["heads"]).to(dev)
        model.load_state_dict(ck["model"])
        model.eval()
        probs = []
        with torch.no_grad():
            for s in range(0, len(states), 512):
                batch = collate(states[s:s + 512], dev)
                for ch in (F_ENDTURN_MEAN, F_ENDTURN_WORST, F_ENDTURN_DEATH, F_ENDTURN_KNOWN):  # trained ablated
                    batch["feats"][:, :, ch] = 0.0
                logits = model(**batch)[1].masked_fill(torch.from_numpy(~legal[s:s + 512]).to(dev), float("-inf"))
                probs.append(torch.softmax(logits, 1).cpu().numpy())
        p = np.concatenate(probs).astype(np.float64)
        top = p.argmax(1)
        exp_reg = best - (p * np.where(legal, cq, 0)).sum(1)
        top_reg = best - cq[np.arange(len(top)), top]
        agree = top == np.where(legal, visits, -1).argmax(1)
        ent = -(np.where(p > 0, p * np.log(np.where(p > 0, p, 1)), 0)).sum(1)
        res["arms"][name] = {"exp_q_regret": float(exp_reg[keep].mean()), "top1_q_regret": float(top_reg[keep].mean()),
                             "top1_most_visited": float(agree[keep].mean()), "entropy": float(ent[keep].mean())}
        print(name, json.dumps(res["arms"][name]), flush=True)
    cqs = {n: v for n, v in res["arms"].items() if n.startswith("arm-cq")}
    res["chosen"] = min(cqs, key=lambda n: cqs[n]["exp_q_regret"]) if cqs else None
    gen_loop.write_json(out / "heldout.json", res)
    print("chosen", res["chosen"])


def _eval_position(idx: int, row: dict, arms: dict[str, str | None], reps: int) -> list[dict]:
    import a3_devset  # noqa: PLC0415
    import combat_prior  # noqa: PLC0415
    import live_search  # noqa: PLC0415
    import search_configs as sc  # noqa: PLC0415
    from combat_tokens import MAX_ACTIONS  # noqa: PLC0415
    from sts2_gym import native  # noqa: PLC0415

    lab, pid = row["label"], row["pid"]
    doc = json.loads((DEVSET / "positions" / f"{pid}.json").read_text(encoding="utf-8"))
    h = native.load_combat_snapshot(doc, seed=7919 * idx + 1)
    out = {"pid": pid, **{k: row[k] for k in ("floor", "kind", "hard", "hp_frac", "n_moves")}, "net": {}, "search": {}}
    try:
        mask = native.valid_actions(h, MAX_ACTIONS)
        legal = [x for x in range(MAX_ACTIONS) if mask[x]]
        sweep = live_search.search_live(h, max_sims=len(legal) + 1, min_child_sims=1, c=0.2, confirm_pairs=0,
                                        time_budget_s=60.0, seed=11, resample=True)
        aims = Counter(ch.action for ch in sweep.children)
        groups = [(ch.action, ch.target, (ch.action, *ch.merged_actions), aims[ch.action]) for ch in sweep.children]
        reg = {(x, t): lab["regret"].get(f"{x},{t}") for x, t, _, _ in groups}
        out["encounter"] = int(native.encounter_id(h))
        pos = sc.Position(kind=row["kind"], n_moves=row["n_moves"], hp_frac=row["hp_frac"], hard=bool(row["hard"]))
        kw = sc.A3_V1.search_kwargs(pos, env_prior=False)
        for name, path in arms.items():
            extra: dict = {}
            if path:
                champ = combat_prior.find(path)
                extra = champ.live_settings()
                p = dict(zip(legal, champ.live_prior()(h, legal).tolist(), strict=True))
                pm = {(x, t): sum(p.get(y, 0.0) for y in acts) / n for x, t, acts, n in groups}
                by_act: dict = {}
                for (x, _t), v in pm.items():
                    by_act[x] = by_act.get(x, 0.0) + v
                top = max(by_act, key=lambda x: (by_act[x], -x))
                known = [k for k in pm if reg[k] is not None]
                tr = [reg[k] for k in pm if k[0] == top and reg[k] is not None]
                mass = sum(pm[k] for k in known)
                out["net"][name] = {"agree": int(top == lab["action"]), "top1_regret": sum(tr) / len(tr) if tr else None,
                                    "exp_regret": sum(pm[k] * reg[k] for k in known) / mass if mass > 0 else None,
                                    "p_label": pm.get((lab["action"], lab["target"]), 0.0), "unknown_mass": 1 - mass}
            got = []
            for rep in range(reps):
                t = time.perf_counter()
                res = live_search.search_live(h, **kw, **extra, seed=1_000_003 * (rep + 1) + 7919 * idx)
                b = res.best
                got.append([rep, int(a3_devset._same_move(b, lab)), lab["regret"].get(f"{b.action},{b.target}"),  # noqa: SLF001
                            round((time.perf_counter() - t) * 1000, 2), res.sims])
            out["search"][name] = got
    finally:
        native.destroy(h)
    return [out]


def devset(a) -> None:
    import a3_devset  # noqa: PLC0415

    out = Path(a.out)
    arms: dict[str, str | None] = {"none": None, **{n: str(d) for n, d in arm_dirs(out).items()}}
    if a.champion:
        arms["champion"] = str(Path(json.loads((DEVSET.parent / "current.json").read_text())["dir"]))
    rows = [r for r in a3_devset.read_jsonl(DEVSET / "labels.jsonl") if "error" not in r][: a.limit or None]
    t0 = time.time()
    got = a3_devset._pool([(i, r, arms, a.reps) for i, r in enumerate(rows)], _eval_position, a.procs,  # noqa: SLF001
                          out / "devset.jsonl")
    print(len(got), "positions", round(time.time() - t0), "s", "errors", sum("error" in g for g in got))


def _mean_or(xs: list, empty: float = 0.0) -> float:
    return float(np.mean(xs)) if xs else empty


def report(a, base: str = BASE) -> dict:
    """Population-weighted means and paired bootstrap intervals of `a.out`/devset.jsonl, each arm against `base`."""
    import a3_devset as ad  # noqa: PLC0415

    out = Path(a.out)
    rows = [r for r in ad.read_jsonl(out / "devset.jsonl") if "error" not in r]
    w = ad._weights(rows, json.loads((DEVSET / "manifest.json").read_text()))  # noqa: SLF001
    arms = list(rows[0]["search"])
    per: dict = {}  # "net/<arm>/<metric>" or "search/<arm>/<metric>" -> {pid: value}
    for n in rows[0]["net"]:
        for m in ("agree", "top1_regret", "exp_regret", "p_label"):
            per[f"net/{n}/{m}"] = {r["pid"]: r["net"][n][m] or 0.0 for r in rows}
    for n in arms:  # a3_devset.regret_by_pos's convention: the mean of the known regrets, 0 if none is known
        per[f"search/{n}/regret"] = {r["pid"]: _mean_or([x[2] for x in r["search"][n] if x[2] is not None]) for r in rows}
        per[f"search/{n}/agree"] = {r["pid"]: _mean_or([x[1] for x in r["search"][n]]) for r in rows}
    doc = {"positions": len(rows), "reps": len(rows[0]["search"][arms[0]]),
           "means": {k: sum(w[p] * v[p] for p in v) / len(v) for k, v in per.items()},
           "ms": {n: _mean_or([x[3] for r in rows for x in r["search"][n]]) for n in arms},
           "sims": {n: _mean_or([x[4] for r in rows for x in r["search"][n]]) for n in arms},
           "search_regret_unknown": {n: sum(x[2] is None for r in rows for x in r["search"][n]) for n in arms},
           "net_unknown_mass": {n: _mean_or([r["net"][n]["unknown_mass"] for r in rows]) for n in rows[0]["net"]},
           "encounter_unknown": sum(int(r.get("encounter", 0)) <= 0 for r in rows), "paired_vs_visits": {}}
    for k in per:
        kind, n, m = k.split("/")
        if n != base and f"{kind}/{base}/{m}" in per:
            doc["paired_vs_visits"][k] = ad.paired_boot(per[k], per[f"{kind}/{base}/{m}"], w)
    gen_loop.write_json(out / "report.json", doc)
    print(json.dumps(doc, indent=1))
    return doc


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("cmd", choices=("train", "heldout", "devset", "report"))
    ap.add_argument("--out", required=True)
    ap.add_argument("--runs", default=str(RUNS))
    ap.add_argument("--gen", type=int, default=3)
    ap.add_argument("--temps", type=lambda s: [float(x) for x in s.split(",")], default=[3.0, 10.0, 30.0])
    ap.add_argument("--data", default="", help="train on this saved train.npz instead of rebuilding the window")
    ap.add_argument("--temps-only", action="store_true", help="train the completed-Q arms only")
    ap.add_argument("--epochs", type=int, default=16)
    ap.add_argument("--device", default="")
    ap.add_argument("--procs", type=int, default=8)
    ap.add_argument("--reps", type=int, default=12)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--champion", action="store_true", help="also search with the promoted generation's prior")
    a = ap.parse_args(argv)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    {"train": train, "heldout": heldout, "devset": devset, "report": report}[a.cmd](a)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
