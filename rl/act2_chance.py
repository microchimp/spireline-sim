#!/usr/bin/env python3
"""A2, the chance of killing the Act 2 boss given the Act 1 boss is killed, and H2, the fight net's forecast of a deck against Act 2's fights
(D59, `referee/backlog/act2-build/OBJECTIVE.md` sections 1, 3 and 4, lane A).

    python rl/act2_chance.py build  --out WORK [--procs 4] [--max-shards N]        (emulator venv: human shards and our self-play runs -> rows)
    python rl/act2_chance.py fit    --rows WORK --init run_value.npz --out OUT     (GPU venv, torch and numpy only: the fit, `a2.pt` and `a2.npz`)
    python rl/act2_chance.py report --rows WORK --model OUT/a2.npz --out OUT       (numpy: AUC, calibration and Look 0 (a) -> report.json, report.md)

H2 (`hive_block`) is the fight block H (`run_train.features.fight_blocks`: E[HP lost] and P(death), the uniform mean over a pool, for the weak, normal,
elite and boss pools) computed with the Hive's pools and its three bosses instead of the row's own region's, from a stored row's base vector alone. Act 2 is
always the Hive (`RunConstants.ActCandidatesByIndex` is [[Overgrowth, Underdocks], [Hive], [Glory]]; D58 addendum 16), so the same eight numbers say what the
deck on this row would face next act. A2 reads the 1,863-number row the student and V read (`models.value_inputs`: 350 dense numbers and the 1,513 counts of
decks, relics and potions) plus H2: V's trunk (the three embeddings and two 256-wide layers), one logit, and a source column (human rows 1, ours 0; the
column is read at ours, exactly the run judgement's `SourceNet`). The score is P(the run kills the Act 2 boss | it kills the Act 1 boss, the state this row
is the afterstate of).

Rows. A row is the chosen option's afterstate row of an Act 1 decision (card, ancient, rest, route and shop screens) of a run that killed the Act 1 boss.
Humans: `decisions-v1` shards with the run's `final` floor and `won`; the target is 1 iff `final` >= 34 or `won`, and a run with `final` < 18 has no rows
(it did not kill the Act 1 boss). Fifths 1 to 3 fit (a tenth of their runs, by a hash of the run, chooses the epoch), fifth 4 is the report set, the exam
fifth is never read: the shards hold fifths 1 to 4 only (the corpus build left the exam out), `check_no_exam` refuses any other fifth, and this module
never opens the corpus folder. Ours: the self-play window of the run judgement (`gen_loop.fighter_manifests`: the blocks one combat prior played, exploring
only), the chosen rows of each run's Act 1 decisions, the target from the run's outcome; a hash of the seed holds out a fifth of the seeds for the report
(and a tenth of the rest chooses the epoch); seeds of any other block (the E-blocks and committed benchmark seeds) are left out. Every run weighs the same
inside its source (`run_weights`).

Look 0 (a): on our held-out runs, the slope of the actual Act 2 boss kill on A2's logit at the Act 1 door (the run's last Act 1 decision row), with a 95 in
100 bootstrap interval over runs; the memo's stop rule is a lower end of 0.5 or less. Nothing here prints a human run id or a seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
import types
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

FORMAT = "act2-chance/1"
HIVE_REGION = 3  # run_train.vocab.REGION_HIVE (a test holds them equal)
ACT1_KILLED_FINAL = 18  # the first floor of Act 2: a run whose final floor is below it did not kill the Act 1 boss
ACT2_KILLED_FINAL = 34  # the first floor of Act 3: a run that ended there or later killed the Act 2 boss
KINDS = ("card", "ancient", "rest", "route", "shop")
TRAIN_FIFTHS = (1, 2, 3)
REPORT_FIFTH = 4
VALID_FIFTHS = (1, 2, 3, 4)  # the shards hold these; the exam bucket (0) is never in them
EXAM_BUCKET = 0
PART = {"train": 0, "val": 1, "report": 2}
VAL_FRACTION = 0.1
OURS_REPORT_FRACTION = 0.2
H2_SIZE = 8  # features.FIGHT_BLOCK_SIZE
DENSE_WIDTH = 350
COUNTS_WIDTH = 1513
BASE_SIZE = 1855
ROW_WIDTH = BASE_SIZE + H2_SIZE  # 1,863: the row V and the student read
E_SIZE, F_SIZE, G_SIZE = 1154, 296, 63
EMBED_WIDTHS = (64, 32, 8)
HIDDEN = 256
INPUT_WIDTH = DENSE_WIDTH + sum(EMBED_WIDTHS) + H2_SIZE  # 462
STOP_SLOPE = 0.5  # the memo's stop rule: the slope's lower bound must be above this
MIN_CELL = 20  # D13: a cell over fewer runs or rows is not shown
DENSE_LIMIT = 1000.0  # a row with a dense number beyond this is not a state (one human record holds 1e8 gold): it is left out, counted
OURS_PRIOR_SHA256 = "64d057e174ac28782a81b353948e87f09b377e84b7403e16e882bc7a45e7be77"  # the combat prior of generations 11 to 13 (the pi_h players)
OURS_GENERATIONS = (13,)
PARAM_KEYS = ("embed.deck.weight", "embed.relics.weight", "embed.potions.weight", "hidden1.weight", "hidden1.bias", "hidden2.weight", "hidden2.bias",
              "score.weight", "score.bias", "src_w")
FIGHT_KEYS = ("embed.deck.weight", "embed.relics.weight", "embed.potions.weight", "hidden1.weight", "hidden1.bias", "hidden2.weight", "hidden2.bias",
              "out.weight", "out.bias")
ROW_KEYS = ("dense", "counts", "y", "src", "run", "floor", "door", "part")  # the arrays every row set holds (a set may also hold h2, seed, fifth)
LAB_RUNS = Path("D:/spireline/referee/LAB/runs")
LAB_SEEDS = Path("D:/spireline/referee/LAB/seeds")
HUMAN_DECISIONS = Path("D:/spireline/referee/human-prior/decisions-v1")
RUN_VALUE = Path("D:/spireline/referee/agent/gen14/candidate-student/run_value.npz")
OUT_DIR = Path("D:/spireline/referee/agent/act2/v1")
WORK_DIR = Path("D:/spireline/logs/act2-a-scratch")


def utc_now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


def file_sha256(path: Path | str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _uniform(salt: str, key: str) -> float:
    """A stable number in [0, 1) from a salt and a key (the split's coin; the key is never written out)."""
    return int.from_bytes(hashlib.sha256(f"{salt}|{key}".encode()).digest()[:8], "big") / 2**64


# ------------------------------------------------------------------------------------------------------------------ targets and exclusions


def target_of(final: int, won: bool) -> int | None:
    """A human run's target: None for a run that ended before floor 18 (it did not kill the Act 1 boss), else 1 iff it ended on floor 34 or later or won."""
    if int(final) < ACT1_KILLED_FINAL and not won:
        return None
    return 1 if (int(final) >= ACT2_KILLED_FINAL or bool(won)) else 0


def ours_target(outcome: dict[str, Any]) -> int | None:
    """One of our runs' target from its recorded outcome, the same rule; None too for a run the clock cut (it has no outcome)."""
    if outcome.get("truncated"):
        return None
    return target_of(int(outcome["final_floor"]), bool(outcome.get("won")))


def check_no_exam(fifths: Iterable[int], run_ids: Sequence[str] | None = None, exam_ids: Iterable[str] | None = None) -> None:
    """Refuse the exam fifth: any fifth outside 1 to 4 (the exam bucket is 0), and with `exam_ids` any run id on that list.

    The shards were built without the exam fifth (fifths 1 to 4 only); this reads the shard's own fifth column, never the corpus folder.
    """
    seen = sorted({int(f) for f in fifths})
    bad = [f for f in seen if f not in VALID_FIFTHS]
    if bad:
        raise ValueError(f"act2_chance: fifths {bad} are not in 1 to 4; the exam fifth (bucket {EXAM_BUCKET}) is never read")
    if exam_ids is not None and run_ids is not None:
        clash = set(map(str, run_ids)) & set(map(str, exam_ids))
        if clash:
            raise ValueError(f"act2_chance: {len(clash)} run ids are in the exam fifth")


def part_of_seed(seed: str) -> int:
    """Our run's part by its seed: a fifth held out for the report, a tenth of the rest chooses the epoch, the rest fits."""
    if _uniform("act2-report", str(seed)) < OURS_REPORT_FRACTION:
        return PART["report"]
    return PART["val"] if _uniform("act2-val", str(seed)) < VAL_FRACTION else PART["train"]


# ------------------------------------------------------------------------------------------------------------------ H2


def hive_queries() -> list[Any]:
    """The fights H2 summarises: the Hive's weak, normal and elite pools and its three bosses (the Hive's boss pool), in `fight_queries`' order."""
    from run_train import features, vocab  # noqa: PLC0415 -- the engine's library loads with it

    pools = vocab.encounter_pools()[vocab.REGION_HIVE]
    return [features.FightQuery(None, int(enc), pool == "weak", pool) for pool in features.FIGHT_POOLS for enc in pools[pool]]  # type: ignore[arg-type]


def hive_block(fight: Any, rows: np.ndarray) -> np.ndarray:
    """H2 of base vectors (1,855 wide) or of rows of the 1,863-wide vector (the base is read, H is not): (n, 8), `fight.summary` on the Hive's queries."""
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] not in (BASE_SIZE, ROW_WIDTH):
        raise ValueError(f"H2 reads base vectors of {BASE_SIZE} numbers or rows of {ROW_WIDTH}, not shape {rows.shape}")
    return fight.summary(np.ascontiguousarray(rows[:, :BASE_SIZE]), hive_queries())


def hive_block_parts(fight: Any, dense: np.ndarray, counts: np.ndarray, chunk: int = 2048) -> np.ndarray:
    """H2 of rows stored as their model inputs (`models.value_inputs`: 350 dense numbers, 1,513 counts): FightNet reads block A, block D and the counts only."""
    from run_train import features  # noqa: PLC0415

    dense = np.asarray(dense)
    counts = np.asarray(counts)
    out = np.zeros((len(dense), H2_SIZE), dtype=np.float32)
    for s in range(0, len(dense), chunk):
        d, c = dense[s:s + chunk].astype(np.float32), counts[s:s + chunk].astype(np.float32)
        base = np.zeros((len(d), BASE_SIZE), dtype=np.float32)
        base[:, features.A_OFFSET:features.A_OFFSET + features.A_SIZE] = d[:, 0:features.A_SIZE]
        base[:, features.D_OFFSET:features.D_OFFSET + features.D_SIZE] = d[:, features.D_OFFSET:features.D_OFFSET + features.D_SIZE]
        base[:, features.E_OFFSET:features.COUNTS_END] = c
        out[s:s + chunk] = hive_block(fight, base)
    return out


def fight_arrays(run_value_npz: Path | str) -> dict[str, np.ndarray]:
    """The FightNet weights of a run judgement's `run_value.npz` (the `fight/` arrays), as `NumpyFightNet` takes them."""
    with np.load(run_value_npz, allow_pickle=False) as z:
        return {k[len("fight/"):]: np.asarray(z[k], dtype=np.float32) for k in z.files if k.startswith("fight/")}


def value_arrays(run_value_npz: Path | str) -> dict[str, np.ndarray]:
    """The value net's weights of a run judgement's `run_value.npz` (the `value/` arrays): the trunk A2 starts from."""
    with np.load(run_value_npz, allow_pickle=False) as z:
        return {k[len("value/"):]: np.asarray(z[k], dtype=np.float32) for k in z.files if k.startswith("value/")}


# ------------------------------------------------------------------------------------------------------------------ rows


def _empty_rows() -> dict[str, list]:
    return {k: [] for k in ("dense", "counts", "y", "src", "run", "floor", "door", "part", "seed", "h2", "fifth")}


def concat_rows(parts: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Join row sets, giving every set's runs numbers of their own (`run` is renumbered past the sets before it); arrays a set lacks are left out of all."""
    parts = [p for p in parts if len(p.get("y", ())) > 0]
    if not parts:
        return {k: np.zeros(0) for k in ROW_KEYS}
    keys = [k for k in parts[0] if all(k in p for p in parts)]
    out: dict[str, np.ndarray] = {}
    offset = 0
    runs = []
    for p in parts:
        runs.append(np.asarray(p["run"], dtype=np.int64) + offset)
        offset += int(np.max(p["run"])) + 1
    for k in keys:
        out[k] = np.concatenate(runs) if k == "run" else np.concatenate([np.asarray(p[k]) for p in parts])
    return out


def select(rows: dict[str, np.ndarray], part: str) -> dict[str, np.ndarray]:
    """The rows of one part ("train", "val" or "report")."""
    mask = rows["part"] == PART[part]
    n = len(rows["y"])
    return {k: v[mask] for k, v in rows.items() if len(v) == n}


def save_rows(path: Path | str, rows: dict[str, np.ndarray]) -> None:
    np.savez_compressed(path, **rows)


def load_rows(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def _door_flags(run: np.ndarray, floor: np.ndarray) -> np.ndarray:
    """True on each run's last Act 1 decision row (the highest floor; the later row among equal floors): the state at the Act 1 door."""
    n = len(run)
    key = floor.astype(np.int64) * 1_000_000 + np.arange(n)
    order = np.lexsort((key, run))
    last = np.r_[run[order][1:] != run[order][:-1], True] if n else np.zeros(0, dtype=bool)
    door = np.zeros(n, dtype=bool)
    door[order[last]] = True
    return door


def human_rows(ds: Any, kinds: Sequence[str] = KINDS, fight: Any = None, stats: dict[str, int] | None = None) -> dict[str, np.ndarray]:
    """The chosen row of every Act 1 decision of `kinds` in the runs of a human `Dataset` (or anything with `dec`, `dense`, `counts`, `run_ids`) that
    killed the Act 1 boss, with its target, run, floor, door flag, fifth and part. With `fight` (the run judgement's FightNet) each row also has H2, computed
    from the row's own numbers (not their float16 store). A row with a dense number past `DENSE_LIMIT` is left out and counted in `stats["outlier_rows"]`.

    Raises:
        ValueError: A fifth outside 1 to 4 (the exam fifth), or shards that carry no true map block (the v0 layout).
    """
    import human_prior as hp  # noqa: PLC0415

    dec = ds.dec
    C = hp.C
    check_no_exam(np.unique(dec[:, C["fifth"]]).tolist(), run_ids=list(ds.run_ids))
    if ds.dense.shape[1] != hp.N_DENSE_V1:
        raise ValueError("act2_chance: the human shards must be the v1 layout (the true map block); decisions-v1 is")
    bad = hp._bool_unavailable(ds, True, True, True, True)  # noqa: SLF001 -- the fit's own filter: replay mismatches and unverified shop prices
    kind_ids = [hp.KIND[k] for k in kinds]
    sel = (~bad) & np.isin(dec[:, C["kind"]], kind_ids) & (dec[:, C["act"]] == 0) & (dec[:, C["final"]] >= ACT1_KILLED_FINAL) & (dec[:, C["chosen"]] >= 0)
    idx = np.flatnonzero(sel)
    row_ix = dec[idx, C["start"]] + dec[idx, C["chosen"]]
    dense32 = hp.assemble_np_v1(ds.dense[row_ix]).astype(np.float32)
    inside = np.abs(dense32).max(axis=1) <= DENSE_LIMIT if len(idx) else np.zeros(0, dtype=bool)
    if stats is not None:
        stats["outlier_rows"] = stats.get("outlier_rows", 0) + int((~inside).sum())
    idx, row_ix, dense32 = idx[inside], row_ix[inside], dense32[inside]
    final, won = dec[idx, C["final"]], dec[idx, C["won"]]
    y = np.where((final >= ACT2_KILLED_FINAL) | (won == 1), 1, 0).astype(np.int8)
    run = dec[idx, C["run"]].astype(np.int64)
    floor = dec[idx, C["floor"]].astype(np.int16)
    fifth = dec[idx, C["fifth"]].astype(np.int8)
    run_ids = list(ds.run_ids)
    part = np.array([PART["report"] if f == REPORT_FIFTH else (PART["val"] if _uniform("act2-val-human", run_ids[r]) < VAL_FRACTION else PART["train"])
                     for f, r in zip(fifth.tolist(), run.tolist())], dtype=np.int8)
    counts = np.asarray(ds.counts[row_ix], dtype=np.uint8)
    out = {"dense": dense32.astype(np.float16), "counts": counts, "y": y, "src": np.ones(len(idx), dtype=np.int8), "run": run, "floor": floor,
           "door": _door_flags(run, floor), "part": part, "fifth": fifth}
    if fight is not None:
        out["h2"] = hive_block_parts(fight, dense32, counts)
    return out


def load_human_rows(dec_dir: Path | str = HUMAN_DECISIONS, fifths: Iterable[int] = VALID_FIFTHS, kinds: Sequence[str] = KINDS, fight: Any = None,
                    stats: dict[str, int] | None = None) -> dict[str, np.ndarray]:
    """`human_rows` over the shards of `dec_dir`, one shard at a time (a shard's whole decision table never joins the rest), keeping the fifths asked for."""
    fifths = {int(f) for f in fifths}
    parts = []
    for p in sorted(Path(dec_dir).glob("shard-*.npz")):
        with np.load(p, allow_pickle=False) as z:
            dec = z["dec"]
            import human_prior as hp  # noqa: PLC0415

            keep_dec = np.isin(dec[:, hp.C["fifth"]], list(fifths | {EXAM_BUCKET}))  # an exam row in a shard is kept here so `human_rows` refuses it
            if not keep_dec.any():
                continue
            ds = types.SimpleNamespace(dec=dec, dense=z["dense"], counts=z["counts"], run_ids=[str(x) for x in z["run_ids"]])
            ds = _only_fifths(ds, fifths | {EXAM_BUCKET}, hp)
            rows = human_rows(ds, kinds, fight, stats)
            parts.append(rows)
    return concat_rows(parts)


def _only_fifths(ds: Any, fifths: set[int], hp: Any) -> Any:
    """The decisions of `ds` in the given fifths, their rows renumbered to follow each other (a `Dataset` of the fifths asked for)."""
    dec = ds.dec
    keep = np.isin(dec[:, hp.C["fifth"]], list(fifths))
    d = dec[keep].copy()
    starts, ns = d[:, hp.C["start"]], d[:, hp.C["n"]]
    sel = np.concatenate([np.arange(s, s + n) for s, n in zip(starts, ns)]) if len(d) else np.zeros(0, dtype=np.int64)
    d[:, hp.C["start"]] = np.concatenate([[0], np.cumsum(ns)[:-1]]) if len(d) else np.zeros(0, dtype=np.int64)
    return types.SimpleNamespace(dec=d, dense=ds.dense[sel], counts=ds.counts[sel], run_ids=ds.run_ids)


def ours_rows(records: Iterable[dict[str, Any]], nets: Any, kinds: Sequence[str] = KINDS, with_h2: bool = True,
              skip_seeds: Iterable[str] = ()) -> dict[str, np.ndarray]:
    """The chosen row of every Act 1 decision of `kinds` in the self-play run records that killed the Act 1 boss (one set of arrays, runs numbered 0..),
    each with its run's target. A decision is the screen as the student sees it (`imitation_chooser.student_kind_of`); `nets` is the run judgement's nets
    (its FightNet fills H, as it did at play). A run with a seed in `skip_seeds` is left out."""
    import imitation_chooser as ic  # noqa: PLC0415
    from human_prior_infer import option_rows  # noqa: PLC0415
    from run_train import models, shards  # noqa: PLC0415

    skip = {str(s) for s in skip_seeds}
    acc = _empty_rows()
    run_no = 0
    for rec in records:
        seed = str(rec["seed"])
        if seed in skip:
            continue
        y = ours_target(rec["outcome"])
        if y is None:
            continue
        rows, floors = [], []
        for entry in rec["decisions"]:
            chosen = entry.get("chosen")
            if chosen is None or int(entry["state"]["act_index"]) != 0:
                continue
            decision = shards.entry_decision(entry)
            if ic.student_kind_of(decision) not in kinds:
                continue
            view = shards.state_view(rec, entry)
            rows.append(option_rows(nets, view, [decision.options[int(chosen)]])[0])
            floors.append(int(entry["state"]["floor"]))
        if not rows:
            continue
        r = np.stack(rows).astype(np.float32)
        dense, counts = models.value_inputs(r)
        if np.abs(dense).max() > DENSE_LIMIT:
            continue  # not a state (never seen in self-play)
        k = len(rows)
        floor = np.array(floors, dtype=np.int16)
        acc["dense"].append(dense.astype(np.float16))
        acc["counts"].append(counts.astype(np.uint8))
        acc["y"].append(np.full(k, y, dtype=np.int8))
        acc["src"].append(np.zeros(k, dtype=np.int8))
        acc["run"].append(np.full(k, run_no, dtype=np.int64))
        acc["floor"].append(floor)
        acc["door"].append(_door_flags(np.zeros(k, dtype=np.int64), floor))
        acc["part"].append(np.full(k, part_of_seed(seed), dtype=np.int8))
        acc["seed"].append(np.full(k, seed))
        if with_h2:
            acc["h2"].append(hive_block(nets.fight, r))
        run_no += 1
    out: dict[str, np.ndarray] = {}
    for key, vals in acc.items():
        if vals:
            out[key] = np.concatenate(vals)
    if "y" not in out:
        return {"dense": np.zeros((0, DENSE_WIDTH), np.float16), "counts": np.zeros((0, COUNTS_WIDTH), np.uint8), "y": np.zeros(0, np.int8),
                "src": np.zeros(0, np.int8), "run": np.zeros(0, np.int64), "floor": np.zeros(0, np.int16), "door": np.zeros(0, bool),
                "part": np.zeros(0, np.int8), "seed": np.zeros(0, "<U1")}
    return out


def with_h2(rows: dict[str, np.ndarray], fight: Any) -> dict[str, np.ndarray]:
    """The rows with H2 added (from their stored dense numbers and counts)."""
    out = dict(rows)
    out["h2"] = hive_block_parts(fight, rows["dense"], rows["counts"])
    return out


def run_weights(run: np.ndarray, src: np.ndarray) -> np.ndarray:
    """Row weights that make every run count once inside its source: 1 / (rows of the run), scaled so each source's mean weight is 1."""
    run = np.asarray(run)
    src = np.asarray(src)
    _, inv, cnt = np.unique(run, return_inverse=True, return_counts=True)
    w = 1.0 / cnt[inv.reshape(-1)]
    out = np.zeros(len(run), dtype=np.float64)
    for s in np.unique(src):
        m = src == s
        out[m] = w[m] / w[m].mean()
    return out.astype(np.float32)


# ------------------------------------------------------------------------------------------------------------------ the net


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * x))


def init_params(value: dict[str, np.ndarray] | None, seed: int = 0) -> dict[str, np.ndarray]:
    """A2's starting weights: the value net's trunk (embeddings, `hidden1` with H2's eight columns appended at zero, `hidden2`), a small `score` head and a
    zero source column. With `value` None every layer is a fresh uniform init (the planted-relation tests)."""
    rng = np.random.default_rng(seed)

    def uni(shape: tuple[int, ...], fan_in: int) -> np.ndarray:
        b = 1.0 / math.sqrt(fan_in)
        return rng.uniform(-b, b, shape).astype(np.float32)

    if value is None:
        p = {"embed.deck.weight": uni((EMBED_WIDTHS[0], E_SIZE), E_SIZE), "embed.relics.weight": uni((EMBED_WIDTHS[1], F_SIZE), F_SIZE),
             "embed.potions.weight": uni((EMBED_WIDTHS[2], G_SIZE), G_SIZE), "hidden1.weight": uni((HIDDEN, INPUT_WIDTH), INPUT_WIDTH),
             "hidden1.bias": uni((HIDDEN,), INPUT_WIDTH), "hidden2.weight": uni((HIDDEN, HIDDEN), HIDDEN), "hidden2.bias": uni((HIDDEN,), HIDDEN)}
    else:
        h1 = np.asarray(value["hidden1.weight"], dtype=np.float32)
        if h1.shape != (HIDDEN, INPUT_WIDTH - H2_SIZE):
            raise ValueError(f"the value net's first layer is {h1.shape}, not {(HIDDEN, INPUT_WIDTH - H2_SIZE)}")
        p = {k: np.asarray(value[k], dtype=np.float32).copy() for k in ("embed.deck.weight", "embed.relics.weight", "embed.potions.weight", "hidden1.bias",
                                                                         "hidden2.weight", "hidden2.bias")}
        p["hidden1.weight"] = np.concatenate([h1, np.zeros((HIDDEN, H2_SIZE), np.float32)], axis=1)
    p["score.weight"] = (uni((1, HIDDEN), HIDDEN) * 0.1).astype(np.float32)
    p["score.bias"] = np.zeros(1, np.float32)
    p["src_w"] = np.zeros(HIDDEN, np.float32)
    return p


def _check_params(params: dict[str, np.ndarray]) -> None:
    missing = [k for k in PARAM_KEYS if k not in params]
    if missing:
        raise ValueError(f"A2's weights lack {missing}")
    want = {"embed.deck.weight": (EMBED_WIDTHS[0], E_SIZE), "embed.relics.weight": (EMBED_WIDTHS[1], F_SIZE), "embed.potions.weight": (EMBED_WIDTHS[2], G_SIZE),
            "hidden1.weight": (HIDDEN, INPUT_WIDTH), "hidden2.weight": (HIDDEN, HIDDEN), "score.weight": (1, HIDDEN), "src_w": (HIDDEN,)}
    wrong = {k: tuple(np.asarray(params[k]).shape) for k, s in want.items() if tuple(np.asarray(params[k]).shape) != s}
    if wrong:
        raise ValueError(f"A2's weights have the wrong shapes: {wrong} (wanted {want})")


def np_logits(params: dict[str, np.ndarray], dense: np.ndarray, counts: np.ndarray, h2: np.ndarray, src: np.ndarray | None = None,
              chunk: int = 8192) -> np.ndarray:
    """A2's logit of rows given as their inputs: dense (n, 350), counts (n, 1,513), H2 (n, 8). `src` (n; 0 ours, 1 human) adds the source column to the
    first layer's pre-activation; None is source 0, ours. Dropout is off. Equal to the torch net's evaluation pass."""
    p = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}
    out = np.zeros(len(dense), dtype=np.float32)
    for s in range(0, len(dense), chunk):
        d = np.asarray(dense[s:s + chunk], dtype=np.float32)
        c = np.asarray(counts[s:s + chunk], dtype=np.float32)
        emb = np.concatenate([c[:, :E_SIZE] @ p["embed.deck.weight"].T, c[:, E_SIZE:E_SIZE + F_SIZE] @ p["embed.relics.weight"].T,
                              c[:, E_SIZE + F_SIZE:] @ p["embed.potions.weight"].T], axis=1)
        x = np.concatenate([d, emb, np.asarray(h2[s:s + chunk], dtype=np.float32)], axis=1)
        pre = x @ p["hidden1.weight"].T + p["hidden1.bias"]
        if src is not None:
            pre = pre + np.asarray(src[s:s + chunk], dtype=np.float32)[:, None] * p["src_w"][None, :]
        h = np.maximum(pre, 0.0)
        h = np.maximum(h @ p["hidden2.weight"].T + p["hidden2.bias"], 0.0)
        out[s:s + chunk] = (h @ p["score.weight"].T + p["score.bias"])[:, 0]
    return out


def _net_class() -> Any:
    import torch  # noqa: PLC0415

    class A2Net(torch.nn.Module):
        """The run judgement's trunk with one logit, H2 beside the dense numbers and the source column (`src_w`) on the first layer."""

        def __init__(self, dropout: float = 0.1) -> None:
            super().__init__()
            self.embed = torch.nn.ModuleDict({"deck": torch.nn.Linear(E_SIZE, EMBED_WIDTHS[0], bias=False),
                                              "relics": torch.nn.Linear(F_SIZE, EMBED_WIDTHS[1], bias=False),
                                              "potions": torch.nn.Linear(G_SIZE, EMBED_WIDTHS[2], bias=False)})
            self.hidden1 = torch.nn.Linear(INPUT_WIDTH, HIDDEN)
            self.hidden2 = torch.nn.Linear(HIDDEN, HIDDEN)
            self.score = torch.nn.Linear(HIDDEN, 1)
            self.src_w = torch.nn.Parameter(torch.zeros(HIDDEN))
            self.dropout = torch.nn.Dropout(dropout)

        def forward(self, dense, counts, h2, src=None):  # noqa: ANN001
            emb = torch.cat([self.embed["deck"](counts[:, :E_SIZE]), self.embed["relics"](counts[:, E_SIZE:E_SIZE + F_SIZE]),
                             self.embed["potions"](counts[:, E_SIZE + F_SIZE:])], dim=1)
            pre = self.hidden1(torch.cat([dense, emb, h2], dim=1))
            if src is not None:
                pre = pre + src[:, None] * self.src_w[None, :]
            h = torch.relu(self.hidden2(self.dropout(torch.relu(pre))))
            return self.score(h)[:, 0]

    return A2Net


def torch_logits(params: dict[str, np.ndarray], dense: np.ndarray, counts: np.ndarray, h2: np.ndarray, src: np.ndarray | None = None) -> np.ndarray:
    """The torch net's evaluation pass over the same inputs as `np_logits` (the tests hold the two equal)."""
    import torch  # noqa: PLC0415

    net = _net_class()(0.0)
    net.load_state_dict({k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in params.items()}, strict=True)
    net.eval()
    with torch.no_grad():
        s = None if src is None else torch.from_numpy(np.asarray(src, dtype=np.float32))
        return net(torch.from_numpy(np.asarray(dense, dtype=np.float32)), torch.from_numpy(np.asarray(counts, dtype=np.float32)),
                   torch.from_numpy(np.asarray(h2, dtype=np.float32)), s).numpy()


def _batch_loss(torch: Any, net: Any, d: dict[str, Any], b: Any) -> tuple[Any, Any]:
    logit = net(d["dense"][b].float(), d["counts"][b].float(), d["h2"][b], d["src"][b])
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logit, d["y"][b], reduction="none")
    return loss, logit


def fit_arrays(train: dict[str, np.ndarray], val: dict[str, np.ndarray], init: dict[str, np.ndarray] | None, epochs: int = 8, batch: int = 512,
               lr: float = 3e-4, weight_decay: float = 1e-2, seed: int = 0, device: str | None = None, dropout: float = 0.1,
               evals_per_epoch: int = 4, patience: int = 8, freeze_trunk: bool = False) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit A2 on `train` (arrays dense, counts, h2, src, y, run) from `init` (`init_params`; None for a fresh net), keeping the weights of the evaluation
    with the lowest weighted log-loss on `val` (`evals_per_epoch` evaluations an epoch; stops after `patience` evaluations without an improvement). Every
    run weighs the same inside its source. With `freeze_trunk` the value net's trunk (embeddings, its columns of the first layer, its biases and the
    second layer) stays as it started and only H2's columns, the source column and the score head learn. Returns the best weights and the history."""
    import torch  # noqa: PLC0415

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    p0 = init if init is not None else init_params(None, seed)
    _check_params(p0)
    net = _net_class()(dropout)
    net.load_state_dict({k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in p0.items()}, strict=True)
    net.to(dev)
    keep_cols = INPUT_WIDTH - H2_SIZE
    frozen_h1 = net.hidden1.weight.detach()[:, :keep_cols].clone() if freeze_trunk else None
    if freeze_trunk:
        for name, prm in net.named_parameters():
            if name.startswith(("embed.", "hidden2.")) or name == "hidden1.bias":
                prm.requires_grad_(False)
    trainable = [q for q in net.parameters() if q.requires_grad]

    def to_dev(rows: dict[str, np.ndarray]) -> dict[str, Any]:
        return {"dense": torch.from_numpy(np.asarray(rows["dense"], dtype=np.float16)).to(dev), "counts": torch.from_numpy(np.asarray(rows["counts"], dtype=np.uint8)).to(dev),
                "h2": torch.from_numpy(np.asarray(rows["h2"], dtype=np.float32)).to(dev), "src": torch.from_numpy(np.asarray(rows["src"], dtype=np.float32)).to(dev),
                "y": torch.from_numpy(np.asarray(rows["y"], dtype=np.float32)).to(dev),
                "w": torch.from_numpy(run_weights(rows["run"], rows["src"])).to(dev)}

    tr, va = to_dev(train), to_dev(val)
    opt = torch.optim.AdamW(trainable, lr=lr, weight_decay=weight_decay)
    n = len(train["y"])
    per_epoch = max(1, math.ceil(n / batch))
    steps = max(1, epochs * per_epoch)
    every = max(1, per_epoch // max(1, evals_per_epoch))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, i / steps))))
    hist: dict[str, Any] = {"evals": [], "epochs": epochs, "batch": batch, "lr": lr, "weight_decay": weight_decay, "dropout": dropout, "seed": seed,
                            "device": str(dev), "rows_train": int(n), "rows_val": int(len(val["y"])), "freeze_trunk": bool(freeze_trunk),
                            "evals_per_epoch": evals_per_epoch}
    best: tuple[float, int, dict[str, np.ndarray]] | None = None

    def state_np() -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy().astype(np.float32).copy() for k, v in net.state_dict().items()}

    def evaluate_val() -> tuple[float, float]:
        net.eval()
        vloss, vw, logits = 0.0, 0.0, []
        with torch.no_grad():
            for i in range(0, len(val["y"]), 4096):
                b = torch.arange(i, min(i + 4096, len(val["y"])), device=dev)
                loss, logit = _batch_loss(torch, net, va, b)
                vloss += float((loss * va["w"][b]).sum())
                vw += float(va["w"][b].sum())
                logits.append(logit.cpu().numpy())
        net.train()
        return vloss / max(vw, 1e-12), (auc(np.asarray(val["y"]), np.concatenate(logits)) if logits else float("nan"))

    step, total, wsum, since_best, stop = 0, 0.0, 0.0, 0, False
    net.train()
    for _epoch in range(epochs):
        perm = torch.from_numpy(rng.permutation(n)).to(dev)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            loss, _ = _batch_loss(torch, net, tr, b)
            w = tr["w"][b]
            obj = (loss * w).sum() / w.sum()
            opt.zero_grad(set_to_none=True)
            obj.backward()
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            opt.step()
            sched.step()
            if frozen_h1 is not None:
                with torch.no_grad():
                    net.hidden1.weight[:, :keep_cols] = frozen_h1
            total += float((loss.detach() * w).sum())
            wsum += float(w.sum())
            step += 1
            if step % every == 0 or step == steps:
                v, v_auc = evaluate_val()
                hist["evals"].append({"step": step, "epoch": round(step / per_epoch, 3), "train_loss": total / max(wsum, 1e-12), "val_loss": v, "val_auc": v_auc})
                total, wsum = 0.0, 0.0
                if best is None or v < best[0]:
                    best, since_best = (v, step, state_np()), 0
                else:
                    since_best += 1
                    if since_best >= patience:
                        stop = True
                        break
        if stop:
            break
    assert best is not None
    hist["best_step"] = best[1]
    hist["best_epoch"] = int((best[1] - 1) // per_epoch)
    hist["best_val_loss"] = best[0]
    hist["stopped_early"] = stop
    return best[2], hist


def average_params(states: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """The mean of fits' weights (fits that started from the same weights: the run judgement's recipe for the kept fit)."""
    return {k: np.mean([np.asarray(st[k], dtype=np.float64) for st in states], axis=0).astype(np.float32) for k in states[0]}


def weighted_log_loss(params: dict[str, np.ndarray], rows: dict[str, np.ndarray]) -> float:
    """The weighted log-loss `fit_arrays` selects by, in numpy: every run counts once inside its source; the source column is read as the rows' own."""
    z = np_logits(params, rows["dense"], rows["counts"], rows["h2"], src=rows["src"]).astype(np.float64)
    y = np.asarray(rows["y"], dtype=np.float64)
    loss = np.maximum(z, 0) - z * y + np.log1p(np.exp(-np.abs(z)))
    w = run_weights(rows["run"], rows["src"]).astype(np.float64)
    return float((loss * w).sum() / w.sum())


# ------------------------------------------------------------------------------------------------------------------ files and the scorer


def write_model(prefix: Path | str, params: dict[str, np.ndarray], fight: dict[str, np.ndarray], meta: dict[str, Any]) -> dict[str, Path]:
    """Write A2 as `<prefix>.pt` (torch, weights only) and `<prefix>.npz` (numpy, for processes without torch): the weights, the FightNet weights H2 is
    computed with, and the meta (format, sizes and the fit's record). Returns {"pt": path, "npz": path}."""
    import torch  # noqa: PLC0415

    _check_params(params)
    missing = [k for k in FIGHT_KEYS if k not in fight]
    if missing:
        raise ValueError(f"the FightNet weights lack {missing}")
    meta = {**meta, "format": FORMAT, "h2_size": H2_SIZE, "input_width": INPUT_WIDTH, "row_width": ROW_WIDTH, "read_at_source": 0}
    text = json.dumps(meta, sort_keys=True)
    pt, npz = Path(str(prefix) + ".pt"), Path(str(prefix) + ".npz")
    pt.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"format": FORMAT, "meta": text, "a2": {k: torch.from_numpy(np.asarray(params[k], dtype=np.float32)) for k in PARAM_KEYS},
                "fight": {k: torch.from_numpy(np.asarray(fight[k], dtype=np.float32)) for k in FIGHT_KEYS}}, pt)
    np.savez(npz, meta=np.array(text), **{f"a2/{k}": np.asarray(params[k], dtype=np.float32) for k in PARAM_KEYS},
             **{f"fight/{k}": np.asarray(fight[k], dtype=np.float32) for k in FIGHT_KEYS})
    return {"pt": pt, "npz": npz}


class Act2Scorer:
    """A2 from its file (`a2.pt` or `a2.npz`): `score(rows)` is P(kills the Act 2 boss | kills the Act 1 boss) of rows of the 1,863-wide vector (the row V
    and the student read), read at source 0 (ours) unless `source` says otherwise. H2 is computed from the file's own FightNet weights. `.sha256` is the
    file's. A `.npz` needs numpy alone; a `.pt` needs torch to be read."""

    def __init__(self, path: Path | str) -> None:
        path = Path(path)
        self.path = path
        self.sha256 = file_sha256(path)
        if path.suffix == ".pt":
            import torch  # noqa: PLC0415

            blob = torch.load(path, map_location="cpu", weights_only=True)
            if not isinstance(blob, dict) or blob.get("format") != FORMAT:
                raise ValueError(f"{path.name} is not an A2 file")
            meta = json.loads(blob["meta"])
            params = {k: v.numpy().astype(np.float32) for k, v in blob["a2"].items()}
            fight = {k: v.numpy().astype(np.float32) for k, v in blob["fight"].items()}
        else:
            with np.load(path, allow_pickle=False) as z:
                if "meta" not in z.files:
                    raise ValueError(f"{path.name} is not an A2 file (no meta)")
                meta = json.loads(str(z["meta"]))
                params = {k[3:]: np.asarray(z[k], dtype=np.float32) for k in z.files if k.startswith("a2/")}
                fight = {k[6:]: np.asarray(z[k], dtype=np.float32) for k in z.files if k.startswith("fight/")}
            if meta.get("format") != FORMAT:
                raise ValueError(f"{path.name} is not an A2 file")
        _check_params(params)
        self.meta, self.params, self._fight_params = meta, params, fight
        self._fight: Any = None

    @property
    def fight(self) -> Any:
        if self._fight is None:
            from run_train import models  # noqa: PLC0415

            self._fight = models.NumpyFightNet(self._fight_params)
        return self._fight

    def logit(self, rows: np.ndarray, source: int = 0, h2: np.ndarray | None = None) -> np.ndarray:
        """A2's logit of rows (n, 1,863). `h2` (n, 8) skips computing H2 when the caller holds it already."""
        from run_train import models  # noqa: PLC0415

        rows = np.asarray(rows, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != ROW_WIDTH:
            raise ValueError(f"A2 scores rows of {ROW_WIDTH} numbers (the row V reads), not shape {rows.shape}")
        if not len(rows):
            return np.zeros(0, dtype=np.float32)
        dense, counts = models.value_inputs(rows)
        h = hive_block(self.fight, rows) if h2 is None else np.asarray(h2, dtype=np.float32)
        return np_logits(self.params, dense, counts, h, src=np.full(len(rows), float(source), dtype=np.float32))

    def score(self, rows: np.ndarray, source: int = 0, h2: np.ndarray | None = None) -> np.ndarray:
        """Probabilities (n,): P(kills the Act 2 boss | kills the Act 1 boss, the state of this row), read at ours."""
        return _sigmoid(self.logit(rows, source, h2)).astype(np.float64)


# ------------------------------------------------------------------------------------------------------------------ metrics


def auc(y: np.ndarray, score: np.ndarray) -> float:
    """The area under the ROC curve (the Mann-Whitney statistic, ties half); NaN if one class is missing."""
    y = np.asarray(y).astype(int)
    score = np.asarray(score, dtype=np.float64)
    n1 = int((y == 1).sum())
    n0 = int((y == 0).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    _, inv, cnt = np.unique(score, return_inverse=True, return_counts=True)
    avg = np.cumsum(cnt) - cnt + (cnt + 1) / 2.0
    ranks = avg[inv.reshape(-1)]
    return float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2.0) / (n1 * n0))


def logistic_slope(x: np.ndarray, y: np.ndarray, weights: np.ndarray | None = None, iters: int = 60) -> dict[str, float]:
    """The logistic regression of y on x (intercept and slope) by Newton steps; `weights` per row (a bootstrap's run multiplicities)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    w = np.ones(len(x)) if weights is None else np.asarray(weights, dtype=np.float64)
    beta = np.zeros(2)
    ridge = 1e-8
    converged = False
    for _ in range(iters):
        z = beta[0] + beta[1] * x
        p = _sigmoid(z)
        g0, g1 = float((w * (y - p)).sum()), float((w * (y - p) * x).sum())
        v = w * p * (1 - p)
        h00, h01, h11 = float(v.sum()) + ridge, float((v * x).sum()), float((v * x * x).sum()) + ridge
        det = h00 * h11 - h01 * h01
        if det <= 0:
            break
        step = np.array([(h11 * g0 - h01 * g1) / det, (h00 * g1 - h01 * g0) / det])
        step = np.clip(step, -5.0, 5.0)
        beta = beta + step
        if float(np.abs(step).max()) < 1e-9:
            converged = True
            break
    return {"intercept": float(beta[0]), "slope": float(beta[1]), "converged": converged}


def slope_interval(x: np.ndarray, y: np.ndarray, run: np.ndarray, reps: int = 2000, seed: int = 0, level: float = 0.95) -> dict[str, Any]:
    """The slope of y on x and its percentile interval over runs: each repetition draws whole runs with replacement (every row of a drawn run comes
    with it). `lower` and `upper` are the two ends of the `level` interval, `lower_one_sided` the 5 in 100 point."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    _, inv = np.unique(np.asarray(run), return_inverse=True)
    inv = inv.reshape(-1)
    n_runs = int(inv.max()) + 1 if len(inv) else 0
    point = logistic_slope(x, y)
    rng = np.random.default_rng(seed)
    slopes = np.zeros(reps)
    for i in range(reps):
        mult = np.bincount(rng.integers(0, n_runs, n_runs), minlength=n_runs)
        slopes[i] = logistic_slope(x, y, mult[inv])["slope"]
    tail = (1 - level) / 2 * 100
    return {"slope": point["slope"], "intercept": point["intercept"], "lower": float(np.percentile(slopes, tail)), "upper": float(np.percentile(slopes, 100 - tail)),
            "lower_one_sided": float(np.percentile(slopes, 5)), "reps": reps, "level": level, "n_runs": n_runs, "n_rows": int(len(x))}


def look0_verdict(lower: float) -> str:
    """The memo's stop rule: "go" only if the slope's lower bound is above 0.5."""
    return "go" if (lower is not None and not math.isnan(lower) and lower > STOP_SLOPE) else "stop"


def calibration(p: np.ndarray, y: np.ndarray, bins: int = 10, min_cell: int = MIN_CELL) -> dict[str, Any]:
    """Equal-width bins of the predicted probability: rows, mean prediction and observed rate (cells under `min_cell` rows are not shown), the expected
    calibration error over all rows, the Brier score and the log-loss."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n = len(p)
    edges = np.linspace(0.0, 1.0, bins + 1)
    ix = np.minimum((p * bins).astype(int), bins - 1)
    cells, ece = [], 0.0
    for b in range(bins):
        m = ix == b
        k = int(m.sum())
        if k:
            ece += k / n * abs(float(p[m].mean()) - float(y[m].mean()))
        shown = k >= min_cell
        cells.append({"lo": float(edges[b]), "hi": float(edges[b + 1]), "n": k if shown else None, "mean_p": float(p[m].mean()) if shown else None,
                      "rate": float(y[m].mean()) if shown else None})
    q = np.clip(p, 1e-9, 1 - 1e-9)
    return {"bins": cells, "ece": float(ece) if n else float("nan"), "brier": float(((p - y) ** 2).mean()) if n else float("nan"),
            "log_loss": float(-(y * np.log(q) + (1 - y) * np.log(1 - q)).mean()) if n else float("nan"),
            "mean_p": float(p.mean()) if n else float("nan"), "rate": float(y.mean()) if n else float("nan")}


def evaluate(logit: np.ndarray, y: np.ndarray, run: np.ndarray, door: np.ndarray, reps: int = 1000, seed: int = 0) -> dict[str, Any]:
    """A2's figures on a set of rows: AUC, calibration and the slope with its run bootstrap, on all rows and on the door rows (one per run)."""
    logit = np.asarray(logit, dtype=np.float64)
    y = np.asarray(y).astype(int)
    run = np.asarray(run)
    door = np.asarray(door, dtype=bool)
    p = _sigmoid(logit)
    out: dict[str, Any] = {"rows": int(len(y)), "runs": int(len(np.unique(run))), "door_runs": int(door.sum()), "kills_door": int(y[door].sum()),
                           "kill_rate_door": float(y[door].mean()) if door.any() else float("nan"), "kill_rate_rows": float(y.mean()) if len(y) else float("nan"),
                           "auc_rows": auc(y, logit), "auc_door": auc(y[door], logit[door]),
                           "calibration_rows": calibration(p, y), "calibration_door": calibration(p[door], y[door])}
    out["slope_rows"] = slope_interval(logit, y, run, reps, seed) if len(y) > 1 else None
    out["slope_door"] = slope_interval(logit[door], y[door], run[door], reps, seed) if door.sum() > 1 else None
    return out


# ------------------------------------------------------------------------------------------------------------------ report


def _cell(n: int) -> int | None:
    return int(n) if n >= MIN_CELL else None


def count_block(rows: dict[str, np.ndarray]) -> dict[str, Any]:
    """Counts of a row set: rows, runs, door rows and the kills among the runs (cells under 20 are not shown)."""
    n = len(rows["y"])
    door = np.asarray(rows["door"], dtype=bool)
    return {"rows": _cell(n), "runs": _cell(len(np.unique(rows["run"])) if n else 0), "kills": _cell(int(rows["y"][door].sum())) if n else None,
            "kill_rate": float(rows["y"][door].mean()) if n and door.any() else None}


def _fmt(x: Any, nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def render_md(report: dict[str, Any]) -> str:
    """The report as markdown (the figures of `report.json`, named)."""
    look = report.get("look0a", {})
    lines = ["# A2: the chance of killing the Act 2 boss given the Act 1 boss is killed (Look 0 (a))", "", f"Written {report.get('written', '')}.", "",
             "## Look 0 (a)", "",
             f"On our held-out runs, at the Act 1 door: the slope of the actual Act 2 boss kill on A2's logit is {_fmt(look.get('slope'))}, "
             f"95 in 100 interval [{_fmt(look.get('lower'))}, {_fmt(look.get('upper'))}] (bootstrap over runs; one-sided 5 in 100 point {_fmt(look.get('lower_one_sided'))}), "
             f"over {look.get('door_runs', 'n/a')} runs that killed the Act 1 boss, {look.get('kills', 'n/a')} of which killed the Act 2 boss. "
             f"Stop rule: the lower end must be above {look.get('stop_rule', STOP_SLOPE)}. Verdict: **{look.get('verdict', 'n/a')}**.", ""]
    for name, ev in (report.get("by_source") or {}).items():
        if isinstance(ev, dict) and "mean_p_at_ours" in ev:
            lines += [f"## {name}", "", f"Mean prediction {_fmt(ev['mean_p_at_ours'])} against {_fmt(ev['rate'])} observed; AUC over all rows {_fmt(ev['auc_rows'])}. "
                      "The gap is what the source column takes: the human rows read as if the player were ours.", ""]
            continue
        if not isinstance(ev, dict) or "auc_door" not in ev:
            continue
        lines += [f"## {name}", "", f"rows {ev['rows']}, runs {ev['runs']}, door runs {ev['door_runs']}, kill rate at the door {_fmt(ev['kill_rate_door'])}", "",
                  f"AUC at the door {_fmt(ev['auc_door'])}, over all rows {_fmt(ev['auc_rows'])}; calibration at the door: expected error "
                  f"{_fmt(ev['calibration_door']['ece'])}, Brier {_fmt(ev['calibration_door']['brier'])}, mean prediction {_fmt(ev['calibration_door']['mean_p'])} "
                  f"against {_fmt(ev['calibration_door']['rate'])} observed.", ""]
        sd = ev.get("slope_door")
        if sd:
            lines += [f"Slope at the door {_fmt(sd['slope'])} [{_fmt(sd['lower'])}, {_fmt(sd['upper'])}]; over all rows "
                      f"{_fmt((ev.get('slope_rows') or {}).get('slope'))}.", ""]
    sha = report.get("sha256", {})
    lines += ["## Files", "", f"a2.pt sha256 {sha.get('pt', '')}", "", f"a2.npz sha256 {sha.get('npz', '')}", ""]
    if report.get("counts"):
        lines += ["## Counts", "", "```", json.dumps(report["counts"], indent=1, sort_keys=True), "```", ""]
    for note in report.get("notes", []):
        lines += [f"- {note}"]
    return "\n".join(lines) + "\n"


# ------------------------------------------------------------------------------------------------------------------ build (emulator venv)


def below_normal() -> None:
    """BelowNormal priority through the Windows priority class (no POSIX call)."""
    try:
        import ctypes  # noqa: PLC0415

        k = ctypes.windll.kernel32  # typed: the untyped call truncates the pseudo handle and fails without an error
        k.GetCurrentProcess.restype = ctypes.c_void_p
        k.SetPriorityClass.argtypes = [ctypes.c_void_p, ctypes.c_uint32]
        k.SetPriorityClass(k.GetCurrentProcess(), 0x4000)
    except (AttributeError, OSError):
        pass


_W: dict[str, Any] = {}


def window_shards(runs_root: Path, prior_sha256: str, gens: Sequence[int]) -> list[Path]:
    """The run shards of the self-play window: the exploring blocks `prior_sha256` played in generations `gens` (the run judgement's window)."""
    import gen_loop  # noqa: PLC0415

    mans = gen_loop.fighter_manifests(runs_root, prior_sha256, list(gens), exploring_only=True)
    return [d / name for d, m in mans for name in m.get("run_shards", [])]


def other_block_seeds() -> set[str]:
    """Seeds of every other block (the E-blocks, the committed benchmark blocks and run_train's held-out seeds): never a row."""
    import gen_loop  # noqa: PLC0415

    sets = gen_loop.known_seed_sets(LAB_SEEDS, None, "")
    return {str(s) for seeds in sets.values() for s in seeds}


def cmd_build(a: argparse.Namespace) -> int:
    import multiprocessing as mp  # noqa: PLC0415

    from spawn_pool import SupervisedPool  # noqa: PLC0415

    if a.procs > 4:
        raise SystemExit("act2_chance: at most 4 processes")
    below_normal()
    work = Path(a.out)
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    from run_train import models  # noqa: PLC0415

    nets = models.load_numpy(a.run_value)
    # humans: decisions-v1, fifths 1 to 4 (the shards carry no exam fifth), then H2 on the stored rows
    h_stats: dict[str, int] = {}
    human = load_human_rows(a.human_decisions, fight=nets.fight, stats=h_stats)
    save_rows(work / "rows-human.npz", human)
    print(f"act2_chance: human rows {len(human['y'])}, runs {len(np.unique(human['run']))} ({time.time() - t0:.0f} s)", flush=True)
    # ours: the self-play window
    skip = sorted(other_block_seeds())
    shard_paths = window_shards(Path(a.runs_root), a.prior_sha256, [int(g) for g in a.gens.split(",")])
    if a.max_shards:
        stride = max(1, len(shard_paths) // a.max_shards)
        shard_paths = shard_paths[::stride][:a.max_shards]
    print(f"act2_chance: {len(shard_paths)} self-play shards, {len(skip)} seeds of other blocks skipped", flush=True)
    results: list[dict] = []
    lost = 0
    pending = list(enumerate(shard_paths))
    retried: set[int] = set()
    by_key = dict(pending)
    pool = SupervisedPool(mp.get_context("spawn"), max(1, a.procs), initializer=below_normal, run_timeout_s=1800.0)
    done = 0
    try:
        while pending or pool.inflight():
            while pending and pool.free():
                k, p = pending.pop(0)
                pool.submit(k, ours_task, (str(p), str(a.run_value), skip))
            pool.wait(0.3)
            for d in pool.collect():
                if d.status == "ok":
                    results.append(d.value)
                    done += 1
                    if done % 100 == 0:
                        print(f"act2_chance: {done}/{len(shard_paths)} shards ({time.time() - t0:.0f} s)", flush=True)
                elif d.key not in retried:
                    retried.add(d.key)
                    pending.append((d.key, by_key[d.key]))
                else:
                    lost += 1
                    print(f"act2_chance: shard {d.key} lost: {d.status} {d.value}", file=sys.stderr)
    finally:
        pool.close()
    ours = concat_rows(results)
    save_rows(work / "rows-ours.npz", ours)
    print(f"act2_chance: our rows {len(ours['y'])}, runs {len(np.unique(ours['run']))}, shards lost {lost} ({time.time() - t0:.0f} s)", flush=True)
    (work / "build.json").write_text(json.dumps({"written": utc_now(), "shards": len(shard_paths), "shards_lost": lost, "skip_seeds": len(skip), "gens": a.gens,
                                                 "prior_sha256": a.prior_sha256, "run_value_sha256": file_sha256(a.run_value),
                                                 "human": count_block(human), "ours": count_block(ours), "human_outlier_rows_left_out": h_stats.get("outlier_rows", 0)}, indent=1), encoding="utf-8")
    return 0 if not lost else 2


def ours_task(path: str, run_value: str, skip: list[str]) -> dict[str, np.ndarray]:
    """A pool task: one shard's rows, the nets loaded once per worker."""
    from run_train import models, shards  # noqa: PLC0415

    if "nets" not in _W:
        _W["nets"] = models.load_numpy(run_value)
        _W["skip"] = set(skip)
    return ours_rows(shards.read_shard(Path(path)), _W["nets"], skip_seeds=_W["skip"])


# ------------------------------------------------------------------------------------------------------------------ fit and report


def cmd_fit(a: argparse.Namespace) -> int:
    work = Path(a.rows)
    human, ours = load_rows(work / "rows-human.npz"), load_rows(work / "rows-ours.npz")
    check_no_exam(np.unique(human["fifth"]).tolist())
    train = concat_rows([select(human, "train"), select(ours, "train")])
    val = concat_rows([select(human, "val"), select(ours, "val")])
    value = value_arrays(a.init)
    seeds = [int(x) for x in str(a.seeds).split(",")]
    states, hists = [], []
    for seed in seeds:
        params, hist = fit_arrays(train, val, init_params(value, seed=seed), epochs=a.epochs, batch=a.batch, lr=a.lr, weight_decay=a.weight_decay, seed=seed,
                                  device=a.device, dropout=a.dropout, evals_per_epoch=a.evals_per_epoch, patience=a.patience, freeze_trunk=a.freeze_trunk)
        states.append(params)
        hists.append(hist)
        print(json.dumps({"seed": seed, **{k: hist[k] for k in ("best_step", "best_epoch", "best_val_loss", "device")}}), flush=True)
    per_seed = [weighted_log_loss(st, val) for st in states]
    mean_loss = weighted_log_loss(average_params(states), val) if len(states) > 1 else per_seed[0]
    if a.keep == "mean" and len(states) > 1:
        kept, kept_loss, how = average_params(states), mean_loss, "the mean of the seeds' weights"
    else:
        best = int(np.argmin(per_seed))
        kept, kept_loss, how = states[best], per_seed[best], f"the seed with the lowest validation loss (seed {seeds[best]})"
    record = {"seeds": seeds, "per_seed_val_loss": per_seed, "mean_of_weights_val_loss": mean_loss, "kept": how, "kept_val_loss": kept_loss, "fits": hists}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"written": utc_now(), "fit": record, "run_value_sha256": file_sha256(a.init), "kinds": list(KINDS),
            "rows": {"train": count_block(train), "val": count_block(val)}}
    files = write_model(out / "a2", kept, fight_arrays(a.init), meta)
    (out / "fit.json").write_text(json.dumps({**meta, "sha256": {k: file_sha256(v) for k, v in files.items()}}, indent=1), encoding="utf-8")
    print(json.dumps({"kept": how, "kept_val_loss": kept_loss, "per_seed_val_loss": per_seed, "mean_of_weights_val_loss": mean_loss}), flush=True)
    return 0


def cmd_report(a: argparse.Namespace) -> int:
    work, out = Path(a.rows), Path(a.out)
    human, ours = load_rows(work / "rows-human.npz"), load_rows(work / "rows-ours.npz")
    scorer = Act2Scorer(a.model)
    p = scorer.params

    def logits(rows: dict[str, np.ndarray], source: int) -> np.ndarray:
        return np_logits(p, rows["dense"], rows["counts"], rows["h2"], src=np.full(len(rows["y"]), float(source)))

    h4, o4 = select(human, "report"), select(ours, "report")
    by_source: dict[str, Any] = {}
    by_source["human fifth 4 (read at the human source)"] = evaluate(logits(h4, 1), h4["y"], h4["run"], h4["door"], a.reps, a.seed)
    by_source["ours, held-out seeds (read at ours)"] = evaluate(logits(o4, 0), o4["y"], o4["run"], o4["door"], a.reps, a.seed)
    h_at_ours = _sigmoid(logits(h4, 0))
    by_source["human fifth 4 read at ours (level only)"] = {"mean_p_at_ours": float(h_at_ours.mean()), "rate": float(h4["y"].mean()),
                                                           "auc_rows": auc(h4["y"], logits(h4, 0))}
    door = o4["door"].astype(bool)
    look = slope_interval(logits(o4, 0)[door], o4["y"][door], o4["run"][door], a.reps, a.seed)
    look0 = {"slope": look["slope"], "intercept": look["intercept"], "lower": look["lower"], "upper": look["upper"], "lower_one_sided": look["lower_one_sided"],
             "level": look["level"], "reps": look["reps"], "verdict": look0_verdict(look["lower"]), "stop_rule": STOP_SLOPE, "door_runs": int(door.sum()),
             "kills": int(o4["y"][door].sum()), "kill_rate": float(o4["y"][door].mean())}
    counts = {"human": {name: count_block(select(human, name)) for name in PART}, "ours": {name: count_block(select(ours, name)) for name in PART}}
    fit_doc = json.loads((Path(a.model).parent / "fit.json").read_text(encoding="utf-8")) if (Path(a.model).parent / "fit.json").is_file() else {}
    pt = Path(a.model).with_suffix(".pt")
    report = {"written": utc_now(), "format": FORMAT, "look0a": look0, "by_source": by_source, "counts": counts, "fit": fit_doc.get("fit"),
              "sha256": {"pt": file_sha256(pt) if pt.is_file() else None, "npz": file_sha256(a.model), "run_value": fit_doc.get("run_value_sha256")},
              "notes": ["Humans: decisions-v1, fifths 1 to 3 fit and fifth 4 reports; the exam fifth is never read (the shards hold fifths 1 to 4 only). Aggregates only; cells under 20 are not shown.",
                        "Ours: the exploring self-play blocks (generation 13, the pi_h players' combat prior) with seeds of other blocks left out; a fifth of the seeds by hash is the report set, a tenth of the rest chooses the weights; the door row is the run's last Act 1 decision row.",
                        "Slope below 1: A2's spread is wider than the outcome's (a logit of x means a chance of sigmoid(intercept + slope x) at the door); the scorer returns the fitted probability, uncalibrated by the report set.",
                        "Our kill rate here (the generation 13 exploring blocks, temperature 0.1 and 2 in 100 random picks) is below the memo's 16 to 21 in 100; the source column is read at ours, so the level is this player's.",
                        "Act2Scorer(a2.npz) needs numpy alone; Act2Scorer(a2.pt) needs torch to read the file. Both score rows of the 1,863-wide vector the student and V read, H2 computed inside from the file's own FightNet weights."]}
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(report, indent=1, default=float), encoding="utf-8")
    (out / "report.md").write_text(render_md(report), encoding="utf-8")
    print(json.dumps(look0))
    return 0


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--out", default=str(WORK_DIR))
    b.add_argument("--procs", type=int, default=4)
    b.add_argument("--max-shards", type=int, default=0, help="a stride sample of the window's shards (0 = all)")
    b.add_argument("--run-value", default=str(RUN_VALUE))
    b.add_argument("--human-decisions", default=str(HUMAN_DECISIONS))
    b.add_argument("--runs-root", default=str(LAB_RUNS))
    b.add_argument("--prior-sha256", default=OURS_PRIOR_SHA256)
    b.add_argument("--gens", default=",".join(map(str, OURS_GENERATIONS)))
    f = sub.add_parser("fit")
    f.add_argument("--rows", default=str(WORK_DIR))
    f.add_argument("--init", default=str(RUN_VALUE))
    f.add_argument("--out", default=str(OUT_DIR))
    f.add_argument("--epochs", type=int, default=8)
    f.add_argument("--batch", type=int, default=512)
    f.add_argument("--lr", type=float, default=3e-4)
    f.add_argument("--weight-decay", type=float, default=1e-2)
    f.add_argument("--seeds", default="0", help="fits from the same start, one per seed; the kept fit is the mean of their weights")
    f.add_argument("--keep", choices=("best", "mean"), default="best", help="the seed with the lowest validation loss, or the mean of the seeds' weights")
    f.add_argument("--device", default=None)
    f.add_argument("--dropout", type=float, default=0.1)
    f.add_argument("--evals-per-epoch", type=int, default=4)
    f.add_argument("--patience", type=int, default=8)
    f.add_argument("--freeze-trunk", action="store_true")
    r = sub.add_parser("report")
    r.add_argument("--rows", default=str(WORK_DIR))
    r.add_argument("--model", default=str(OUT_DIR / "a2.npz"))
    r.add_argument("--out", default=str(OUT_DIR))
    r.add_argument("--reps", type=int, default=2000)
    r.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)
    return {"build": cmd_build, "fit": cmd_fit, "report": cmd_report}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
