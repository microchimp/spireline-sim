#!/usr/bin/env python3
"""The boss chain: two forecasts that give the run chooser's option values a sight of the Act 2 boss and the first Act 3 boss (Act 3 week, D68 item 3;
`referee/backlog/act3-week/REFIT-WIN-DESIGN.md` sections 2 and 5, lane 3b).

    q2 = P(the run gets past the Act 2 boss)                              (rows of Act 2 decisions)
    q3 = P(it gets past the first Act 3 boss | it got past the Act 2 boss)   (rows of Act 2 and Act 3 decisions)

The run brain (`run_value.npz`) forecasts the final floor and cannot tell door positions apart (AUC 0.59 for the first Act 3 boss among the runs that
reach it). The chain is a small second network of the same trunk, fitted on the plain-run stores (`referee/run-cheap/<gen>/`, generations 15 to 18) to the
runs' own outcomes, with no share of any forecast. The chooser (`imitation_chooser.BlendedPolicy`, agent.json block `boss_chain`) adds
`kappa_act * log(chance of getting through)` to each option's value in Acts 2 and 3: `log q2 + log q3` up to the Act 2 boss's floor, `log q3` after it up
to floor 47. Act 1 is never touched (kappa 0). The recorded values stay the run brain's.

    python rl/boss_chain.py build  --stores 15,16,17,18 --run-value NPZ --out DIR [--procs 4]   (emulator venv: the stores -> rows.npz, build.json)
    python rl/boss_chain.py fit    --rows DIR --init NPZ --out DIR [--seeds 0,1,2]               (GPU venv: boss_chain.npz, .pt, fit.json)
    python rl/boss_chain.py report --rows DIR --model NPZ --out DIR [--champion]                  (numpy: AUC, calibration, the verdict)
    python rl/boss_chain.py flips  --champion DIR --candidate DIR --store DIR --out JSON          (emulator venv: what the chain changes on stored runs)

Rows. One row is the chosen option's afterstate row of a decision in Act 2 (floors up to 32, runs past the Act 1 boss, at most 12 a run) or Act 3
(floors up to 47), as `human_prior_infer.option_rows` builds it (the run brain's row with the FightNet's H of the row's own region), kept as its model
inputs (`models.value_inputs`) plus H3, the FightNet's forecast against Act 3's pools and its three first and last bosses (`region_block`). Every run's rows
weigh 1 inside a head (`run_weights_masked`). Generation 18's runs, a fifth of them by a hash of the seed, are the report set; they are never fitted on.
Nothing here writes or prints a seed.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
import time
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import act2_chance as a2  # noqa: E402 -- numpy only

FORMAT = "boss-chain/1"
STORES = (15, 16, 17, 18)
STORE_ROOT = Path("D:/spireline/referee/run-cheap")
READ_GEN = 18  # the generation whose offset the player reads: the candidate plays this generation's fight brain
PASS_B1, PASS_B2, PASS_B3A, WON = 18, 34, 49, 53  # the final floor at or above which a run got past the Act 1 boss, the Act 2 boss, the first Act 3 boss; a win
LAST_PRE_B2, LAST_PRE_B3A = 32, 47  # the last decision floors before the Act 2 boss and the first Act 3 boss
ACT2_ROWS_PER_RUN = 12
CLIP = 0.002  # a chance is clipped below at this before its log
KAPPA = (0.0, 10.0, 10.0)  # floors per nat in Acts 1, 2 and 3
MAX_KAPPA = 50.0
H3_SIZE = 8
GLORY_REGION = 4  # run_train.vocab.REGION_GLORY: Act 3 is always that region (a test holds them equal)
ERA_BUDGET, ERA_ASCENSION = "30", 10  # the plain runs' data era (train._in_era)
REPORT_FRACTION, VAL_FRACTION = 0.2, 0.1
PART = a2.PART
MIN_CELL = a2.MIN_CELL
DENSE_LIMIT = a2.DENSE_LIMIT
DENSE_WIDTH, COUNTS_WIDTH, ROW_WIDTH, BASE_SIZE = a2.DENSE_WIDTH, a2.COUNTS_WIDTH, a2.ROW_WIDTH, a2.BASE_SIZE
E_SIZE, F_SIZE, G_SIZE, EMBED_WIDTHS, HIDDEN = a2.E_SIZE, a2.F_SIZE, a2.G_SIZE, a2.EMBED_WIDTHS, a2.HIDDEN
INPUT_WIDTH = DENSE_WIDTH + sum(EMBED_WIDTHS) + H3_SIZE  # 462
PARAM_KEYS = ("embed.deck.weight", "embed.relics.weight", "embed.potions.weight", "hidden1.weight", "hidden1.bias", "hidden2.weight", "hidden2.bias",
              "head.weight", "head.bias", "gen_off")
FIGHT_KEYS = a2.FIGHT_KEYS
ROW_KEYS = ("dense", "counts", "h3", "gen", "run", "floor", "act", "y2", "m2", "y3", "m3", "entry2", "entry3", "door", "reached48", "won", "pv", "part")
DOOR_PASS_AUC, DOOR_STOP_AUC = 0.70, 0.65
SLOPE_BAND = (0.7, 1.3)
MEAN_BAND = 0.03
RUN_VALUE = Path("D:/spireline/referee/agent/gen18/candidate-fight-v3/run_value.npz")
LOGS = Path("D:/spireline/logs/boss-chain")

utc_now = a2.utc_now
file_sha256 = a2.file_sha256
below_normal = a2.below_normal
auc = a2.auc


# ------------------------------------------------------------------------------------------------------------------ outcomes, parts, H3


def passed(final_floor: int, won: bool = False) -> tuple[bool, bool, bool, bool]:
    """(past the Act 1 boss, past the Act 2 boss, past the first Act 3 boss, won) of a run from its final floor (53 means won; `won` says it too)."""
    f = int(final_floor)
    w = bool(won) or f >= WON
    return w or f >= PASS_B1, w or f >= PASS_B2, w or f >= PASS_B3A, w


def part_of(gen: int, seed: str) -> int:
    """A run's part: a fifth of generation 18's runs by a hash of the seed is the report set (never another generation's), a tenth of the rest chooses the
    epoch, the rest fits."""
    if int(gen) == READ_GEN and a2._uniform("boss-chain-report", str(seed)) < REPORT_FRACTION:  # noqa: SLF001
        return PART["report"]
    return PART["val"] if a2._uniform("boss-chain-val", str(seed)) < VAL_FRACTION else PART["train"]  # noqa: SLF001


def region_queries(region: int) -> list[Any]:
    """The fights H summarises for a region: its weak, normal and elite pools and its boss pool (all its bosses), in `fight_queries`' order."""
    from run_train import features, vocab  # noqa: PLC0415 -- the engine's library loads with it

    pools = vocab.encounter_pools()[int(region)]
    return [features.FightQuery(None, int(enc), pool == "weak", pool) for pool in features.FIGHT_POOLS for enc in pools[pool]]  # type: ignore[arg-type]


def region_block(fight: Any, rows: np.ndarray, region: int) -> np.ndarray:
    """H of base vectors (1,855 wide) or rows of the 1,863-wide vector (the base is read, H is not) computed with `region`'s pools and bosses: (n, 8)."""
    rows = np.asarray(rows, dtype=np.float32)
    if rows.ndim != 2 or rows.shape[1] not in (BASE_SIZE, ROW_WIDTH):
        raise ValueError(f"H reads base vectors of {BASE_SIZE} numbers or rows of {ROW_WIDTH}, not shape {rows.shape}")
    return fight.summary(np.ascontiguousarray(rows[:, :BASE_SIZE]), region_queries(region))


def fight_arrays_sha256(arrays: dict[str, np.ndarray]) -> str:
    """The sha256 of FightNet arrays (keys in `FIGHT_KEYS` order: name, shape, float32 bytes): what a chain file records as `fight_sha256`."""
    h = hashlib.sha256()
    for k in FIGHT_KEYS:
        a = np.ascontiguousarray(np.asarray(arrays[k], dtype=np.float32))
        h.update(k.encode())
        h.update(str(a.shape).encode())
        h.update(a.tobytes())
    return h.hexdigest()


def fight_sha_of(run_value_npz: Path | str) -> str:
    """The `fight_sha256` of a run judgement's `run_value.npz` (its `fight/` arrays), computed as `write_model` computes it."""
    return fight_arrays_sha256(a2.fight_arrays(run_value_npz))


# ------------------------------------------------------------------------------------------------------------------ rows


def _chosen_scorable(entry: dict[str, Any]) -> bool:
    """`train._chosen_scorable` (a test holds them equal; `train` imports torch, which the workers never load)."""
    return bool(entry["options"][entry["chosen"]].get("scorable", True))


def in_era(rec: dict[str, Any]) -> bool:
    """`train._in_era` for the plain runs' era: combat budget, ascension and feature version."""
    from run_train import FEATURE_VERSION  # noqa: PLC0415

    return (rec.get("combat", {}).get("budget") == ERA_BUDGET and rec.get("ascension") == ERA_ASCENSION and rec.get("feature_version") == FEATURE_VERSION)


def _cap_pick(seed: str, indices: Sequence[int], keep: int) -> list[int]:
    """`keep` of `indices` chosen by a hash of (seed, decision index); the first index is always among them (the state at the start of the act)."""
    if len(indices) <= keep:
        return list(indices)
    rest = sorted(indices[1:], key=lambda i: hashlib.sha256(f"boss-chain-cap|{seed}|{i}".encode()).digest())[: keep - 1]
    return sorted([indices[0], *rest])


def empty_rows() -> dict[str, np.ndarray]:
    return {"dense": np.zeros((0, DENSE_WIDTH), np.float16), "counts": np.zeros((0, COUNTS_WIDTH), np.uint8), "h3": np.zeros((0, H3_SIZE), np.float32),
            "gen": np.zeros(0, np.int8), "run": np.zeros(0, np.int64), "floor": np.zeros(0, np.int16), "act": np.zeros(0, np.int8),
            "y2": np.zeros(0, np.int8), "m2": np.zeros(0, np.int8), "y3": np.zeros(0, np.int8), "m3": np.zeros(0, np.int8),
            "entry2": np.zeros(0, bool), "entry3": np.zeros(0, bool), "door": np.zeros(0, bool), "reached48": np.zeros(0, bool), "won": np.zeros(0, np.int8),
            "pv": np.zeros((0, 2), np.float32), "part": np.zeros(0, np.int8)}


def chain_rows(records: Iterable[dict[str, Any]], nets: Any, skip_seeds: Iterable[str] = (), stats: dict[str, int] | None = None,
               with_keys: bool = False) -> dict[str, np.ndarray]:
    """The chain's rows of self-play run records (one set of arrays, runs numbered from 0).

    A record counts when it is of generation `STORES`' latest or earlier, was not truncated, is of the plain runs' era, and its seed is not another block's
    (`skip_seeds`). Its decisions with a chosen, scorable option give rows: Act 2 ones (`act_index` 1, floor up to 32, runs past the Act 1 boss; at most
    `ACT2_ROWS_PER_RUN` a run, the first always, the rest by a hash of the seed and the decision's index) and Act 3 ones (`act_index` 2, floor up to 47, all).
    With `with_keys` the set also has `run_key` (a hash of (generation, seed) per run, for the caller's copy check; never written to a file).
    """
    from human_prior_infer import option_rows  # noqa: PLC0415
    from run_train import models, shards  # noqa: PLC0415

    skip = {str(s) for s in skip_seeds}
    st = stats if stats is not None else {}
    parts: dict[str, list[np.ndarray]] = {k: [] for k in ROW_KEYS}
    keys: list[int] = []
    run_no = 0
    for rec in records:
        st["records"] = st.get("records", 0) + 1
        seed = str(rec["seed"])
        gen = rec.get("generation")
        if seed in skip:
            st["skipped_other_block"] = st.get("skipped_other_block", 0) + 1
            continue
        if gen is None or int(gen) not in STORES:
            st["dropped_generation"] = st.get("dropped_generation", 0) + 1
            continue
        if rec["outcome"].get("truncated"):
            st["dropped_truncated"] = st.get("dropped_truncated", 0) + 1
            continue
        if not in_era(rec):
            st["dropped_era"] = st.get("dropped_era", 0) + 1
            continue
        final, won_flag = int(rec["outcome"]["final_floor"]), bool(rec["outcome"].get("won"))
        b1, b2, b3a, won = passed(final, won_flag)
        act2, act3 = [], []
        for i, entry in enumerate(rec["decisions"]):
            if entry.get("chosen") is None or not _chosen_scorable(entry):
                continue
            s = entry["state"]
            act, floor = int(s["act_index"]), int(s["floor"])
            if act == 1 and floor <= LAST_PRE_B2 and b1:
                act2.append(i)
            elif act == 2 and floor <= LAST_PRE_B3A and b2:
                act3.append(i)
        chosen_ix = _cap_pick(seed, act2, ACT2_ROWS_PER_RUN) + act3
        if not chosen_ix:
            continue
        rows, acts, floors = [], [], []
        for i in chosen_ix:
            entry = rec["decisions"][i]
            decision = shards.entry_decision(entry)
            view = shards.state_view(rec, entry)
            rows.append(option_rows(nets, view, [decision.options[int(entry["chosen"])]])[0])
            acts.append(int(entry["state"]["act_index"]))
            floors.append(int(entry["state"]["floor"]))
        r = np.stack(rows).astype(np.float32)
        dense, counts = models.value_inputs(r)
        ok = np.abs(dense).max(axis=1) <= DENSE_LIMIT
        if not ok.all():  # not a state (never seen in self-play)
            st["outlier_rows"] = st.get("outlier_rows", 0) + int((~ok).sum())
        if not ok.any():
            continue
        r, dense, counts = r[ok], dense[ok], counts[ok]
        act = np.array(acts, dtype=np.int8)[ok]
        floor = np.array(floors, dtype=np.int16)[ok]
        h3 = region_block(nets.fight, r, GLORY_REGION)
        p = np.asarray(nets.value.floor_probs(r), dtype=np.float64)
        pv = np.stack([p[:, PASS_B2 - 1:].sum(axis=1), p[:, PASS_B3A - 1:].sum(axis=1)], axis=1).astype(np.float32)  # P(final >= 34), P(final >= 49)
        k = len(r)
        is2 = act == 1
        is3 = act == 2
        entry2 = np.zeros(k, bool)
        entry3 = np.zeros(k, bool)
        door = np.zeros(k, bool)
        if is2.any():
            entry2[int(np.flatnonzero(is2)[0])] = True
        if is3.any():
            entry3[int(np.flatnonzero(is3)[0])] = True
            door[int(np.flatnonzero(is3)[-1])] = True
        parts["dense"].append(dense.astype(np.float16))
        parts["counts"].append(counts.astype(np.uint8))
        parts["h3"].append(h3.astype(np.float32))
        parts["gen"].append(np.full(k, int(gen), np.int8))
        parts["run"].append(np.full(k, run_no, np.int64))
        parts["floor"].append(floor)
        parts["act"].append(act)
        parts["y2"].append(np.full(k, int(b2), np.int8))
        parts["m2"].append(is2.astype(np.int8))
        parts["y3"].append(np.full(k, int(b3a), np.int8))
        parts["m3"].append((is3 | (is2 & b2)).astype(np.int8))
        parts["entry2"].append(entry2)
        parts["entry3"].append(entry3)
        parts["door"].append(door)
        parts["reached48"].append(np.full(k, final >= 48 or won, bool))
        parts["won"].append(np.full(k, int(won), np.int8))
        parts["pv"].append(pv)
        parts["part"].append(np.full(k, part_of(int(gen), seed), np.int8))
        keys.append(int.from_bytes(hashlib.sha256(f"{int(gen)}|{seed}".encode()).digest()[:8], "big"))
        run_no += 1
        st["runs"] = st.get("runs", 0) + 1
    out = {k: np.concatenate(v) for k, v in parts.items() if v}
    if not out:
        out = empty_rows()
    if with_keys:
        out["run_key"] = np.array(keys, dtype=np.uint64)
    return out


def concat_rows(sets: Sequence[dict[str, np.ndarray]]) -> dict[str, np.ndarray]:
    """Join row sets in order, numbering each set's runs past the sets before it. With `run_key` in every set a run whose key an earlier set holds is left
    out (a resumed block can hold copies of a run); the survivors' runs are numbered 0.. in order, and the keys are dropped from the result."""
    sets = [s for s in sets if len(s.get("y2", ())) > 0]
    if not sets:
        return empty_rows()
    runs, offset = [], 0
    for s in sets:
        runs.append(np.asarray(s["run"], dtype=np.int64) + offset)
        offset += int(np.max(s["run"])) + 1
    out = {k: np.concatenate([np.asarray(s[k]) for s in sets]) for k in ROW_KEYS if k != "run"}
    run = np.concatenate(runs)
    if all("run_key" in s for s in sets):
        run_keys = np.concatenate([np.asarray(s["run_key"], dtype=np.uint64) for s in sets])  # indexed by global run number (sets' runs are 0..n-1 in order)
        _, first = np.unique(run_keys, return_index=True)
        keep_run = np.zeros(len(run_keys), bool)
        keep_run[first] = True
        keep = keep_run[run]
        if not keep.all():
            out = {k: v[keep] for k, v in out.items()}
            run = run[keep]
    _, dense_run = np.unique(run, return_inverse=True)
    out["run"] = dense_run.reshape(-1).astype(np.int64)
    return {k: out[k] for k in ROW_KEYS}


def select(rows: dict[str, np.ndarray], part: str) -> dict[str, np.ndarray]:
    """The rows of one part ("train", "val" or "report")."""
    mask = rows["part"] == PART[part]
    n = len(rows["y2"])
    return {k: v[mask] for k, v in rows.items() if len(v) == n}


def run_weights_masked(run: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Row weights of one head: each run's masked rows weigh 1 / (its rows in the head), scaled to a mean of 1 over the head's masked rows; 0 off the mask."""
    run = np.asarray(run)
    m = np.asarray(mask).astype(bool)
    w = np.zeros(len(run), dtype=np.float32)
    if m.any():
        _, inv, cnt = np.unique(run[m], return_inverse=True, return_counts=True)
        ww = 1.0 / cnt[inv.reshape(-1)]
        w[m] = (ww / ww.mean()).astype(np.float32)
    return w


def q4_of(rows: dict[str, np.ndarray]) -> dict[str, Any]:
    """The constant forecast of winning given past the first Act 3 boss: (wins + 1) / (runs past it + 2) over the runs the rows hold. Recorded, never used."""
    m3 = np.asarray(rows["m3"]).astype(bool)
    run = np.asarray(rows["run"])[m3]
    y3 = np.asarray(rows["y3"])[m3]
    won = np.asarray(rows["won"])[m3]
    runs, first = np.unique(run, return_index=True)
    past = int(y3[first].sum())
    wins = int(won[first].sum())
    return {"runs_past_b3a": past, "wins": wins, "q4": (wins + 1) / (past + 2)}


def counts_block(rows: dict[str, np.ndarray]) -> dict[str, Any]:
    """Rows, runs and positives per generation and head (no seeds)."""
    out: dict[str, Any] = {}
    gen = np.asarray(rows["gen"])
    for g in STORES:
        gm = gen == g
        cell: dict[str, Any] = {}
        for head, m, y in (("q2", "m2", "y2"), ("q3", "m3", "y3")):
            mm = gm & (np.asarray(rows[m]) > 0)
            runs, first = np.unique(np.asarray(rows["run"])[mm], return_index=True)
            cell[head] = {"rows": int(mm.sum()), "runs": int(len(runs)), "positives": int(np.asarray(rows[y])[mm][first].sum()) if len(runs) else 0}
        out[str(g)] = cell
    return out


# ------------------------------------------------------------------------------------------------------------------ the net


def _sigmoid(x: np.ndarray) -> np.ndarray:
    return 0.5 * (1.0 + np.tanh(0.5 * np.asarray(x, dtype=np.float64)))


def gen_index(gen: np.ndarray) -> np.ndarray:
    """The offset column of each row's generation (its position in `STORES`)."""
    g = np.asarray(gen)
    out = np.full(g.shape, -1, dtype=np.int64)
    for k, s in enumerate(STORES):
        out[g == s] = k
    if (out < 0).any():
        raise ValueError(f"a row has a generation outside {STORES}")
    return out


def init_params(value: dict[str, np.ndarray] | None, seed: int = 0) -> dict[str, np.ndarray]:
    """The chain's starting weights: the run brain's trunk (embeddings, `hidden1` with H3's eight columns appended at zero, `hidden2`), a small two-output
    head and zero offsets per generation. With `value` None every layer is a fresh uniform init (the planted-relation tests)."""
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
        if h1.shape != (HIDDEN, INPUT_WIDTH - H3_SIZE):
            raise ValueError(f"the run brain's first layer is {h1.shape}, not {(HIDDEN, INPUT_WIDTH - H3_SIZE)}")
        p = {k: np.asarray(value[k], dtype=np.float32).copy() for k in ("embed.deck.weight", "embed.relics.weight", "embed.potions.weight", "hidden1.bias",
                                                                         "hidden2.weight", "hidden2.bias")}
        p["hidden1.weight"] = np.concatenate([h1, np.zeros((HIDDEN, H3_SIZE), np.float32)], axis=1)
    p["head.weight"] = (uni((2, HIDDEN), HIDDEN) * 0.1).astype(np.float32)
    p["head.bias"] = np.zeros(2, np.float32)
    p["gen_off"] = np.zeros((2, len(STORES)), np.float32)
    return p


def check_params(params: dict[str, np.ndarray]) -> None:
    missing = [k for k in PARAM_KEYS if k not in params]
    if missing:
        raise ValueError(f"the chain's weights lack {missing}")
    want = {"embed.deck.weight": (EMBED_WIDTHS[0], E_SIZE), "embed.relics.weight": (EMBED_WIDTHS[1], F_SIZE), "embed.potions.weight": (EMBED_WIDTHS[2], G_SIZE),
            "hidden1.weight": (HIDDEN, INPUT_WIDTH), "hidden2.weight": (HIDDEN, HIDDEN), "head.weight": (2, HIDDEN), "head.bias": (2,),
            "gen_off": (2, len(STORES))}
    wrong = {k: tuple(np.asarray(params[k]).shape) for k, s in want.items() if tuple(np.asarray(params[k]).shape) != s}
    if wrong:
        raise ValueError(f"the chain's weights have the wrong shapes: {wrong} (wanted {want})")


def np_logits(params: dict[str, np.ndarray], dense: np.ndarray, counts: np.ndarray, h3: np.ndarray, gen: np.ndarray, chunk: int = 8192) -> np.ndarray:
    """The chain's two logits (q2, q3) of rows given as their inputs: dense (n, 350), counts (n, 1,513), H3 (n, 8) and each row's generation (the offset
    column read). Dropout is off. Equal to the torch net's evaluation pass."""
    p = {k: np.asarray(v, dtype=np.float32) for k, v in params.items()}
    gidx = gen_index(gen)
    out = np.zeros((len(dense), 2), dtype=np.float32)
    for s in range(0, len(dense), chunk):
        d = np.asarray(dense[s:s + chunk], dtype=np.float32)
        c = np.asarray(counts[s:s + chunk], dtype=np.float32)
        emb = np.concatenate([c[:, :E_SIZE] @ p["embed.deck.weight"].T, c[:, E_SIZE:E_SIZE + F_SIZE] @ p["embed.relics.weight"].T,
                              c[:, E_SIZE + F_SIZE:] @ p["embed.potions.weight"].T], axis=1)
        x = np.concatenate([d, emb, np.asarray(h3[s:s + chunk], dtype=np.float32)], axis=1)
        h = np.maximum(x @ p["hidden1.weight"].T + p["hidden1.bias"], 0.0)
        h = np.maximum(h @ p["hidden2.weight"].T + p["hidden2.bias"], 0.0)
        out[s:s + chunk] = h @ p["head.weight"].T + p["head.bias"] + p["gen_off"][:, gidx[s:s + chunk]].T
    return out


def _net_class() -> Any:
    import torch  # noqa: PLC0415

    class ChainNet(torch.nn.Module):
        """The run brain's trunk with two logits, H3 beside the dense numbers, and an offset per head and generation."""

        def __init__(self, dropout: float = 0.3) -> None:
            super().__init__()
            self.embed = torch.nn.ModuleDict({"deck": torch.nn.Linear(E_SIZE, EMBED_WIDTHS[0], bias=False),
                                              "relics": torch.nn.Linear(F_SIZE, EMBED_WIDTHS[1], bias=False),
                                              "potions": torch.nn.Linear(G_SIZE, EMBED_WIDTHS[2], bias=False)})
            self.hidden1 = torch.nn.Linear(INPUT_WIDTH, HIDDEN)
            self.hidden2 = torch.nn.Linear(HIDDEN, HIDDEN)
            self.head = torch.nn.Linear(HIDDEN, 2)
            self.gen_off = torch.nn.Parameter(torch.zeros(2, len(STORES)))
            self.dropout = torch.nn.Dropout(dropout)

        def forward(self, dense, counts, h3, gidx):  # noqa: ANN001
            emb = torch.cat([self.embed["deck"](counts[:, :E_SIZE]), self.embed["relics"](counts[:, E_SIZE:E_SIZE + F_SIZE]),
                             self.embed["potions"](counts[:, E_SIZE + F_SIZE:])], dim=1)
            pre = self.hidden1(torch.cat([dense, emb, h3], dim=1))
            h = torch.relu(self.hidden2(self.dropout(torch.relu(pre))))
            return self.head(h) + self.gen_off[:, gidx].T

    return ChainNet


def torch_logits(params: dict[str, np.ndarray], dense: np.ndarray, counts: np.ndarray, h3: np.ndarray, gen: np.ndarray) -> np.ndarray:
    """The torch net's evaluation pass over the same inputs as `np_logits` (the tests hold the two equal)."""
    import torch  # noqa: PLC0415

    net = _net_class()(0.0)
    net.load_state_dict({k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in params.items()}, strict=True)
    net.eval()
    with torch.no_grad():
        return net(torch.from_numpy(np.asarray(dense, dtype=np.float32)), torch.from_numpy(np.asarray(counts, dtype=np.float32)),
                   torch.from_numpy(np.asarray(h3, dtype=np.float32)), torch.from_numpy(gen_index(gen))).numpy()


def weighted_loss(params: dict[str, np.ndarray], rows: dict[str, np.ndarray]) -> dict[str, float]:
    """The validation loss `fit_arrays` selects by, in numpy: each head's weighted log-loss over its masked rows (every run once inside a head) and their sum."""
    z = np_logits(params, rows["dense"], rows["counts"], rows["h3"], rows["gen"]).astype(np.float64)
    out: dict[str, float] = {}
    for h, (m, y) in enumerate((("m2", "y2"), ("m3", "y3"))):
        w = run_weights_masked(rows["run"], rows[m]).astype(np.float64)
        t = np.asarray(rows[y], dtype=np.float64)
        loss = np.maximum(z[:, h], 0) - z[:, h] * t + np.log1p(np.exp(-np.abs(z[:, h])))
        out[f"q{h + 2}"] = float((loss * w).sum() / w.sum()) if w.sum() > 0 else float("nan")
    out["total"] = sum(v for v in out.values() if not math.isnan(v))
    return out


def fit_arrays(train: dict[str, np.ndarray], val: dict[str, np.ndarray], init: dict[str, np.ndarray] | None, epochs: int = 4, batch: int = 512,
               lr: float = 1e-4, weight_decay: float = 1e-2, seed: int = 0, device: str | None = None, dropout: float = 0.3, evals_per_epoch: int = 4,
               patience: int = 8) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Fit the chain on `train` from `init` (`init_params`; None for a fresh net), keeping the weights of the evaluation with the lowest validation loss
    (the two heads' weighted log-losses summed; `evals_per_epoch` evaluations an epoch; stops after `patience` evaluations without an improvement).
    The loss is a masked BCE: each head's mean over its own rows with every run counting once (`run_weights_masked`), the two summed. Cosine schedule."""
    import torch  # noqa: PLC0415

    dev = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    p0 = init if init is not None else init_params(None, seed)
    check_params(p0)
    net = _net_class()(dropout)
    net.load_state_dict({k: torch.from_numpy(np.asarray(v, dtype=np.float32)) for k, v in p0.items()}, strict=True)
    net.to(dev)

    def to_dev(rows: dict[str, np.ndarray]) -> dict[str, Any]:
        f32 = lambda a: torch.from_numpy(np.asarray(a, dtype=np.float32)).to(dev)  # noqa: E731
        return {"dense": torch.from_numpy(np.asarray(rows["dense"], dtype=np.float16)).to(dev), "counts": torch.from_numpy(np.asarray(rows["counts"], dtype=np.uint8)).to(dev),
                "h3": f32(rows["h3"]), "g": torch.from_numpy(gen_index(rows["gen"])).to(dev), "y2": f32(rows["y2"]), "y3": f32(rows["y3"]),
                "w2": f32(run_weights_masked(rows["run"], rows["m2"])), "w3": f32(run_weights_masked(rows["run"], rows["m3"]))}

    tr, va = to_dev(train), to_dev(val)
    opt = torch.optim.AdamW(net.parameters(), lr=lr, weight_decay=weight_decay)
    n = len(train["y2"])
    per_epoch = max(1, math.ceil(n / batch))
    steps = max(1, epochs * per_epoch)
    every = max(1, per_epoch // max(1, evals_per_epoch))
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * min(1.0, i / steps))))
    hist: dict[str, Any] = {"evals": [], "epochs": epochs, "batch": batch, "lr": lr, "weight_decay": weight_decay, "dropout": dropout, "seed": seed,
                            "device": str(dev), "rows_train": int(n), "rows_val": int(len(val["y2"])), "evals_per_epoch": evals_per_epoch}
    bce = torch.nn.functional.binary_cross_entropy_with_logits
    best: tuple[float, int, dict[str, np.ndarray]] | None = None

    def state_np() -> dict[str, np.ndarray]:
        return {k: v.detach().cpu().numpy().astype(np.float32).copy() for k, v in net.state_dict().items()}

    def head_loss(d: dict[str, Any], b: Any, logit: Any) -> tuple[Any, Any]:
        l2 = (bce(logit[:, 0], d["y2"][b], reduction="none") * d["w2"][b]).sum() / d["w2"][b].sum().clamp_min(1e-9)
        l3 = (bce(logit[:, 1], d["y3"][b], reduction="none") * d["w3"][b]).sum() / d["w3"][b].sum().clamp_min(1e-9)
        return l2, l3

    def evaluate_val() -> tuple[float, float, float, float, float]:
        net.eval()
        sums = np.zeros(4)  # weighted loss of head 2 and 3, weight sums
        logits = []
        with torch.no_grad():
            for i in range(0, len(val["y2"]), 4096):
                b = torch.arange(i, min(i + 4096, len(val["y2"])), device=dev)
                logit = net(va["dense"][b].float(), va["counts"][b].float(), va["h3"][b], va["g"][b])
                sums[0] += float((bce(logit[:, 0], va["y2"][b], reduction="none") * va["w2"][b]).sum())
                sums[1] += float((bce(logit[:, 1], va["y3"][b], reduction="none") * va["w3"][b]).sum())
                sums[2] += float(va["w2"][b].sum())
                sums[3] += float(va["w3"][b].sum())
                logits.append(logit.cpu().numpy())
        net.train()
        z = np.concatenate(logits) if logits else np.zeros((0, 2))
        m2, m3 = np.asarray(val["m2"]).astype(bool), np.asarray(val["m3"]).astype(bool)
        v2, v3 = sums[0] / max(sums[2], 1e-12), sums[1] / max(sums[3], 1e-12)
        return v2 + v3, v2, v3, auc(np.asarray(val["y2"])[m2], z[m2, 0]), auc(np.asarray(val["y3"])[m3], z[m3, 1])

    step, t2, t3, count, since_best, stop = 0, 0.0, 0.0, 0, 0, False
    net.train()
    for _epoch in range(epochs):
        perm = torch.from_numpy(rng.permutation(n)).to(dev)
        for i in range(0, n, batch):
            b = perm[i:i + batch]
            logit = net(tr["dense"][b].float(), tr["counts"][b].float(), tr["h3"][b], tr["g"][b])
            l2, l3 = head_loss(tr, b, logit)
            opt.zero_grad(set_to_none=True)
            (l2 + l3).backward()
            torch.nn.utils.clip_grad_norm_(net.parameters(), 1.0)
            opt.step()
            sched.step()
            t2 += float(l2.detach())
            t3 += float(l3.detach())
            count += 1
            step += 1
            if step % every == 0 or step == steps:
                v, v2, v3, a2c, a3c = evaluate_val()
                hist["evals"].append({"step": step, "epoch": round(step / per_epoch, 3), "train_loss_q2": t2 / max(count, 1), "train_loss_q3": t3 / max(count, 1),
                                      "val_loss": v, "val_loss_q2": v2, "val_loss_q3": v3, "val_auc_q2": a2c, "val_auc_q3": a3c})
                t2, t3, count = 0.0, 0.0, 0
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
    hist["best_is_first_eval"] = bool(hist["evals"] and best[1] == hist["evals"][0]["step"])
    hist["stopped_early"] = stop
    return best[2], hist


# ------------------------------------------------------------------------------------------------------------------ files and the scorer


def write_model(prefix: Path | str, params: dict[str, np.ndarray], fight: dict[str, np.ndarray], meta: dict[str, Any]) -> dict[str, Path]:
    """Write the chain as `<prefix>.npz` (numpy, for the players) and `<prefix>.pt` (torch): the weights, the FightNet arrays H3 is computed with (and
    their sha256, which the agent loader checks against its own run brain's), and the meta (format, sizes, generations, q4, counts and the fit's record).
    Returns {"npz": path, "pt": path}."""
    import torch  # noqa: PLC0415

    check_params(params)
    missing = [k for k in FIGHT_KEYS if k not in fight]
    if missing:
        raise ValueError(f"the FightNet weights lack {missing}")
    meta = {**meta, "format": FORMAT, "h3_size": H3_SIZE, "input_width": INPUT_WIDTH, "row_width": ROW_WIDTH, "read_gen": READ_GEN, "gens": list(STORES),
            "fight_sha256": fight_arrays_sha256(fight)}
    text = json.dumps(meta, sort_keys=True)
    npz, pt = Path(str(prefix) + ".npz"), Path(str(prefix) + ".pt")
    npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez(npz, meta=np.array(text), **{f"chain/{k}": np.asarray(params[k], dtype=np.float32) for k in PARAM_KEYS},
             **{f"fight/{k}": np.asarray(fight[k], dtype=np.float32) for k in FIGHT_KEYS})
    torch.save({"format": FORMAT, "meta": text, "chain": {k: torch.from_numpy(np.asarray(params[k], dtype=np.float32)) for k in PARAM_KEYS},
                "fight": {k: torch.from_numpy(np.asarray(fight[k], dtype=np.float32)) for k in FIGHT_KEYS}}, pt)
    return {"npz": npz, "pt": pt}


class ChainScorer:
    """The chain from its `.npz`, numpy only. `.sha256` is the file's, `.fight_sha256` that of the FightNet arrays it carries (recomputed, and held equal to the
    meta's); `log_q(rows)` is the log of the clipped chances (q2, q3) of rows of the 1,863-wide vector the run brain reads, H3 computed inside from the file's
    own FightNet and the generation offset read at `READ_GEN`."""

    def __init__(self, path: Path | str) -> None:
        path = Path(path)
        self.path = path
        self.sha256 = file_sha256(path)
        with np.load(path, allow_pickle=False) as z:
            if "meta" not in z.files:
                raise ValueError(f"{path.name} is not a boss chain file (no meta)")
            meta = json.loads(str(z["meta"]))
            params = {k[len("chain/"):]: np.asarray(z[k], dtype=np.float32) for k in z.files if k.startswith("chain/")}
            fight = {k[len("fight/"):]: np.asarray(z[k], dtype=np.float32) for k in z.files if k.startswith("fight/")}
        if meta.get("format") != FORMAT:
            raise ValueError(f"{path.name} is not a {FORMAT} file")
        check_params(params)
        self.fight_sha256 = fight_arrays_sha256(fight)
        if meta.get("fight_sha256") != self.fight_sha256:
            raise ValueError(f"{path.name}: its FightNet arrays are not the ones its meta names")
        self.meta, self.params, self._fight_params = meta, params, fight
        self._fight: Any = None

    @property
    def fight(self) -> Any:
        if self._fight is None:
            from run_train import models  # noqa: PLC0415

            self._fight = models.NumpyFightNet(self._fight_params)
        return self._fight

    def logits(self, rows: np.ndarray, h3: np.ndarray | None = None) -> np.ndarray:
        """The two logits of rows (n, 1,863), the generation offset read at `READ_GEN`; `h3` (n, 8) skips computing H3 when the caller holds it."""
        from run_train import models  # noqa: PLC0415

        rows = np.asarray(rows, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[1] != ROW_WIDTH:
            raise ValueError(f"the chain scores rows of {ROW_WIDTH} numbers (the row the run brain reads), not shape {rows.shape}")
        if not len(rows):
            return np.zeros((0, 2), dtype=np.float32)
        dense, counts = models.value_inputs(rows)
        h = region_block(self.fight, rows, GLORY_REGION) if h3 is None else np.asarray(h3, dtype=np.float32)
        return np_logits(self.params, dense, counts, h, np.full(len(rows), READ_GEN, dtype=np.int8))

    def log_q(self, rows: np.ndarray, h3: np.ndarray | None = None) -> np.ndarray:
        """log(clip(sigmoid(logit), CLIP, 1)) of (q2, q3): (n, 2)."""
        return np.log(np.clip(_sigmoid(self.logits(rows, h3)), CLIP, 1.0))


def chain_terms(log_q: np.ndarray, act_index: int, floor: int, kappa: Sequence[float]) -> np.ndarray | None:
    """What the chain adds to each option's value on a screen: None where it adds nothing (kappa 0 for the act, Act 1, an act outside 1 to 2, a floor past
    47); in Act 2 up to the Act 2 boss's floor kappa x (log q2 + log q3); after that kappa x log q3."""
    act_index, floor = int(act_index), int(floor)
    if act_index not in (1, 2) or floor > LAST_PRE_B3A:
        return None
    k = float(kappa[act_index])
    if k == 0:
        return None
    lq = np.asarray(log_q, dtype=np.float64)
    if act_index == 1 and floor <= LAST_PRE_B2:
        return k * (lq[:, 0] + lq[:, 1])
    return k * lq[:, 1]


# ------------------------------------------------------------------------------------------------------------------ build (emulator venv)

_W: dict[str, Any] = {}


def store_shards(stores: Sequence[int], root: Path | str = STORE_ROOT) -> list[tuple[int, Path]]:
    """(store, shard file) of every run shard of the complete jobs of the stores (`gen_loop.find_manifests`: a job counts when its shards.json exists)."""
    import gen_loop  # noqa: PLC0415

    out = []
    for g in stores:
        for d, m in gen_loop.find_manifests(Path(root) / str(g) / "jobs"):
            out += [(int(g), d / name) for name in m.get("run_shards", [])]
    return out


def chain_task(path: str, run_value: str, skip: list[str]) -> tuple[dict[str, np.ndarray], dict[str, int]]:
    """A pool task: one shard's rows (and the counts of what was seen and dropped), the nets loaded once per worker."""
    from run_train import models, shards  # noqa: PLC0415

    if "nets" not in _W:
        _W["nets"] = models.load_numpy(run_value)
        _W["skip"] = set(skip)
    stats: dict[str, int] = {}
    rows = chain_rows(shards.read_shard(Path(path)), _W["nets"], skip_seeds=_W["skip"], stats=stats, with_keys=True)
    return rows, stats


def cmd_build(a: argparse.Namespace) -> int:
    import multiprocessing as mp  # noqa: PLC0415

    from spawn_pool import SupervisedPool  # noqa: PLC0415

    if a.procs > 4:
        raise SystemExit("boss_chain: at most 4 processes")
    below_normal()
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    stores = [int(x) for x in str(a.stores).split(",")]
    tasks = store_shards(stores, a.store_root)
    if a.max_shards:
        stride = max(1, len(tasks) // a.max_shards)
        tasks = tasks[::stride][:a.max_shards]
    skip = sorted(a2.other_block_seeds())
    print(f"boss_chain: {len(tasks)} shards of stores {stores}, {len(skip)} seeds of other blocks skipped", flush=True)
    results: dict[int, tuple[dict[str, np.ndarray], dict[str, int]]] = {}
    lost = 0
    pending = list(enumerate(tasks))
    retried: set[int] = set()
    by_key = dict(pending)
    pool = SupervisedPool(mp.get_context("spawn"), max(1, a.procs), initializer=below_normal, run_timeout_s=1800.0)
    try:
        while pending or pool.inflight():
            while pending and pool.free():
                k, (_g, p) = pending.pop(0)
                pool.submit(k, chain_task, (str(p), str(a.run_value), skip))
            pool.wait(0.3)
            for d in pool.collect():
                if d.status == "ok":
                    results[d.key] = d.value
                    if len(results) % 100 == 0:
                        print(f"boss_chain: {len(results)}/{len(tasks)} shards ({time.time() - t0:.0f} s)", flush=True)
                elif d.key not in retried:
                    retried.add(d.key)
                    pending.append((d.key, by_key[d.key]))
                else:
                    lost += 1
                    print(f"boss_chain: shard {d.key} lost: {d.status} {d.value}", file=sys.stderr)
    finally:
        pool.close()
    ordered = [results[k] for k in sorted(results)]
    stats: dict[str, int] = {}
    for _rows, st in ordered:
        for key, v in st.items():
            stats[key] = stats.get(key, 0) + int(v)
    rows = concat_rows([r for r, _ in ordered])
    dup = stats.get("runs", 0) - len(np.unique(rows["run"]))
    np.savez(out / "rows.npz", **rows)
    build = {"written": utc_now(), "format": FORMAT, "stores": stores, "shards": len(tasks), "shards_lost": lost, "skip_seeds": len(skip), "records": stats,
             "duplicate_runs_dropped": int(dup), "rows": int(len(rows["y2"])), "runs": int(len(np.unique(rows["run"]))), "by_generation": counts_block(rows),
             "q4": q4_of(rows), "run_value_sha256": file_sha256(a.run_value), "acts2_rows_per_run": ACT2_ROWS_PER_RUN,
             "parts": {name: int((rows["part"] == v).sum()) for name, v in PART.items()}, "wall_s": round(time.time() - t0, 1)}
    (out / "build.json").write_text(json.dumps(build, indent=1), encoding="utf-8")
    print(json.dumps({k: build[k] for k in ("shards", "shards_lost", "rows", "runs", "duplicate_runs_dropped", "wall_s")}), flush=True)
    return 0 if not lost else 2


# ------------------------------------------------------------------------------------------------------------------ fit


def load_rows(path: Path | str) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as z:
        return {k: z[k] for k in z.files}


def cmd_fit(a: argparse.Namespace) -> int:
    below_normal()
    work = Path(a.rows)
    rows = load_rows(work / "rows.npz")
    train, val = select(rows, "train"), select(rows, "val")
    value = a2.value_arrays(a.init)
    seeds = [int(x) for x in str(a.seeds).split(",")]
    states, hists = [], []
    for seed in seeds:
        params, hist = fit_arrays(train, val, init_params(value, seed=seed), epochs=a.epochs, batch=a.batch, lr=a.lr, weight_decay=a.weight_decay, seed=seed,
                                  device=a.device, dropout=a.dropout, evals_per_epoch=a.evals_per_epoch, patience=a.patience)
        states.append(params)
        hists.append(hist)
        print(json.dumps({"seed": seed, **{k: hist[k] for k in ("best_step", "best_epoch", "best_val_loss", "best_is_first_eval", "device")}}), flush=True)
    per_seed = [weighted_loss(st, val) for st in states]
    best = int(np.argmin([p["total"] for p in per_seed]))
    kept = states[best]
    record = {"seeds": seeds, "per_seed_val_loss": per_seed, "kept_seed": seeds[best], "kept": f"the seed with the lowest validation loss (seed {seeds[best]})",
              "kept_val_loss": per_seed[best], "fits": hists}
    build_doc = json.loads((work / "build.json").read_text(encoding="utf-8")) if (work / "build.json").is_file() else {}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    meta = {"written": utc_now(), "fit": record, "run_value_sha256": file_sha256(a.init), "q4": build_doc.get("q4") or q4_of(rows),
            "counts": build_doc.get("by_generation") or counts_block(rows), "kappa_default": list(KAPPA), "clip": CLIP,
            "rows": {"train": int(len(train["y2"])), "val": int(len(val["y2"]))}}
    files = write_model(out / "boss_chain", kept, a2.fight_arrays(a.init), meta)
    fit_doc = {**meta, "sha256": {k: file_sha256(v) for k, v in files.items()}}
    (out / "fit.json").write_text(json.dumps(fit_doc, indent=1), encoding="utf-8")
    print(json.dumps({"kept_seed": seeds[best], "kept_val_loss": per_seed[best], "best_is_first_eval": [h["best_is_first_eval"] for h in hists]}), flush=True)
    return 0


# ------------------------------------------------------------------------------------------------------------------ report


def _logit(p: np.ndarray) -> np.ndarray:
    q = np.clip(np.asarray(p, dtype=np.float64), 1e-6, 1 - 1e-6)
    return np.log(q / (1 - q))


def cell_metrics(p: np.ndarray, y: np.ndarray, min_cell: int = MIN_CELL) -> dict[str, Any]:
    """The figures of one forecast against one outcome: rows, mean forecast, actual rate, AUC, the logistic slope of the outcome on the forecast's logit and
    the forecast by decile (equal-count; a decile under `min_cell` rows is not shown). A cell under `min_cell` rows shows its row count and nothing else."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y).astype(int)
    n = len(p)
    if n < min_cell:
        return {"rows": None if n else 0}
    order = np.argsort(p, kind="stable")
    deciles = []
    for chunk in np.array_split(order, 10):
        k = len(chunk)
        deciles.append({"rows": k if k >= min_cell else None, "mean_p": float(p[chunk].mean()) if k >= min_cell else None,
                        "rate": float(y[chunk].mean()) if k >= min_cell else None})
    sl = a2.logistic_slope(_logit(p), y)
    return {"rows": n, "mean_p": float(p.mean()), "rate": float(y.mean()), "positives": int(y.sum()), "auc": auc(y, p), "slope": sl["slope"],
            "intercept": sl["intercept"], "deciles": deciles}


def verdict_of(door: dict[str, Any]) -> dict[str, Any]:
    """The pre-registered smoke test on the chain's door cell: pass when the door AUC is at least 0.70, the slope between 0.7 and 1.3 and the mean forecast
    within 3 points of the actual rate; `band` is go (AUC at least 0.70), between (0.65 to 0.70: build, say so) or stop (under 0.65)."""
    a_, s, m, r = door.get("auc"), door.get("slope"), door.get("mean_p"), door.get("rate")
    if a_ is None or s is None or m is None or r is None or any(isinstance(x, float) and math.isnan(x) for x in (a_, s, m, r)):
        return {"verdict": "no-pass", "band": "stop", "why": "the door cell is too small or one-sided"}
    band = "go" if a_ >= DOOR_PASS_AUC else "between" if a_ >= DOOR_STOP_AUC else "stop"
    ok = a_ >= DOOR_PASS_AUC and SLOPE_BAND[0] <= s <= SLOPE_BAND[1] and abs(m - r) <= MEAN_BAND
    return {"verdict": "pass" if ok else "no-pass", "band": band, "door_auc": a_, "door_slope": s, "door_mean": m, "door_rate": r,
            "rule": f"door AUC >= {DOOR_PASS_AUC}, slope in {list(SLOPE_BAND)}, mean within {MEAN_BAND} of actual; stop under {DOOR_STOP_AUC}"}


def report_cells(rows: dict[str, np.ndarray], model: dict[str, np.ndarray], champion: bool) -> dict[str, Any]:
    """The report set's cells, the chain's forecast and (with `champion`) the run brain's own beside it: q3 at the first Act 3 decision (`entry3`) and at the
    door among the runs that reached the first Act 3 boss, q2 and q2 x q3 at the first Act 2 decision (`entry2`)."""
    z = np_logits(model, rows["dense"], rows["counts"], rows["h3"], rows["gen"]).astype(np.float64)
    q2, q3 = _sigmoid(z[:, 0]), _sigmoid(z[:, 1])
    y2, y3 = np.asarray(rows["y2"]), np.asarray(rows["y3"])
    pv = np.asarray(rows["pv"], dtype=np.float64)
    sel = {"q3_entry3": np.asarray(rows["entry3"], bool), "q3_door": np.asarray(rows["door"], bool) & np.asarray(rows["reached48"], bool),
           "q2_entry2": np.asarray(rows["entry2"], bool), "q2q3_entry2": np.asarray(rows["entry2"], bool)}
    chain = {"q3_entry3": (q3, y3), "q3_door": (q3, y3), "q2_entry2": (q2, y2), "q2q3_entry2": (q2 * q3, y3)}
    champ = {"q3_entry3": (pv[:, 1], y3), "q3_door": (pv[:, 1], y3), "q2_entry2": (pv[:, 0], y2), "q2q3_entry2": (pv[:, 1], y3)}
    out: dict[str, Any] = {}
    for name, m in sel.items():
        cell: dict[str, Any] = {"runs": int(m.sum()), "chain": cell_metrics(chain[name][0][m], chain[name][1][m])}
        if champion:
            cell["champion"] = cell_metrics(champ[name][0][m], champ[name][1][m])
        out[name] = cell
    return out


def _fmt(x: Any, nd: int = 3) -> str:
    return "n/a" if x is None or (isinstance(x, float) and math.isnan(x)) else f"{x:.{nd}f}" if isinstance(x, float) else str(x)


def render_md(report: dict[str, Any]) -> str:
    v = report["verdict"]
    lines = ["# The boss chain: smoke report on the held-out generation 18 runs", "", f"Written {report['written']}. Model sha256 {report['model_sha256'][:12]}.", "",
             f"**Verdict: {v['verdict']}** (door AUC band: {v['band']}). Rule: {v.get('rule', '')}", "",
             "| Cell | Runs | Forecast | Mean forecast | Actual | AUC | Slope |", "|---|---|---|---|---|---|---|"]
    for name, cell in report["cells"].items():
        for who in ("chain", "champion"):
            c = cell.get(who)
            if c is None:
                continue
            lines.append(f"| {name} | {c.get('rows')} | {who} | {_fmt(c.get('mean_p'))} | {_fmt(c.get('rate'))} | {_fmt(c.get('auc'))} | {_fmt(c.get('slope'))} |")
    lines += ["", "Calibration by decile of the forecast (door cell, chain): rows, mean forecast, actual rate.", ""]
    for d in report["cells"]["q3_door"]["chain"].get("deciles", []):
        lines.append(f"- {d['rows']} rows, forecast {_fmt(d['mean_p'])}, actual {_fmt(d['rate'])}")
    return "\n".join(lines) + "\n"


def cmd_report(a: argparse.Namespace) -> int:
    rows = load_rows(Path(a.rows) / "rows.npz")
    rep = select(rows, "report")
    if len(rep["y2"]) and set(np.unique(rep["gen"]).tolist()) - {READ_GEN}:
        raise ValueError("the report set holds a generation other than the read generation")
    scorer_params = ChainScorer(a.model).params
    cells = report_cells(rep, scorer_params, bool(a.champion))
    doc = {"written": utc_now(), "format": FORMAT, "model_sha256": file_sha256(a.model), "report_rows": int(len(rep["y2"])),
           "report_runs": int(len(np.unique(rep["run"]))), "cells": cells, "verdict": verdict_of(cells["q3_door"]["chain"]),
           "notes": ["The report set is a fifth of generation 18's plain runs by a hash of the seed; no row of it was fitted on.",
                     "Door: the last Act 3 decision of a run that reached the first Act 3 boss. Entry: the first decision of the act.",
                     "The run brain's forecast of the same cell is its own P(final floor at least 34) (q2) or at least 49 (q3 and q2 x q3), recomputed from the champion's nets on the same row."]}
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "report.json").write_text(json.dumps(doc, indent=1, default=float), encoding="utf-8")
    (out / "report.md").write_text(render_md(doc), encoding="utf-8")
    print(json.dumps({"verdict": doc["verdict"], "door": {w: {k: cells["q3_door"].get(w, {}).get(k) for k in ("rows", "mean_p", "rate", "auc", "slope")} for w in ("chain", "champion")},
                      "entry3": {w: {k: cells["q3_entry3"].get(w, {}).get(k) for k in ("rows", "mean_p", "rate", "auc")} for w in ("chain", "champion")}}, default=float))
    return 0


# ------------------------------------------------------------------------------------------------------------------ flips (emulator venv)


def _pick_stats() -> dict[str, Any]:
    return {"decisions": 0, "eligible": 0, "changed": 0, "champion_agrees": 0}


def flips(champion_dir: Path | str, candidate_dir: Path | str, store: Path | str, runs: int = 400) -> dict[str, Any]:
    """What the chain changes on stored runs: for generation 18 runs of the report set, every decision the run brain chooses by (two or more options, all
    scored) is picked by the champion and by the candidate from the stored view; counts by act and screen kind of the decisions, the share the candidate
    picks otherwise, and the champion's own agreement with the recorded pick (the check that the replay is faithful). Also the Act 3 rest sites' heal share
    below 70 in 100 of max HP, the Act 3 shops' potion share of purchases and the elite share of the Act 2 and 3 route screens, for each agent."""
    import gen_loop  # noqa: PLC0415
    from imitation_chooser import KIND_OF_TYPE  # noqa: PLC0415
    from run_train import shards, vocab  # noqa: PLC0415

    champ, cand = (gen_loop.load_agent(champion_dir).macro_policy(), gen_loop.load_agent(candidate_dir).macro_policy())
    store = Path(store)
    seeds: list[str] = []
    with open(store / "runs.jsonl", encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                d = json.loads(line)
            except ValueError:
                continue
            if part_of(READ_GEN, str(d["seed"])) == PART["report"]:
                seeds.append(str(d["seed"]))
    seeds = seeds[:runs]
    by: dict[str, dict[str, Any]] = {}
    side = {"rest_low_hp": {"screens": 0, "champion_heals": 0, "candidate_heals": 0},
            "shop_act3": {"purchases_champion": 0, "potions_champion": 0, "purchases_candidate": 0, "potions_candidate": 0},
            "route_act23": {"screens": 0, "champion_elite": 0, "candidate_elite": 0}}
    done = 0
    for seed in seeds:
        path = store / "decisions.d" / f"{seed}.record.json"
        if not path.is_file():
            continue
        rec = json.loads(path.read_text(encoding="utf-8"))
        champ.begin_run(seed)
        cand.begin_run(seed)
        for entry in rec["decisions"]:
            if entry.get("chosen") is None:
                continue
            act = int(entry["state"]["act_index"])
            kind = KIND_OF_TYPE.get(entry["type"], entry["type"])
            cell = by.setdefault(f"act{act + 1}/{kind}", _pick_stats())
            cell["decisions"] += 1
            decision = shards.entry_decision(entry)
            options = decision.options
            if len(options) < 2 or any(not o.scorable for o in options):
                continue
            if act >= 1 and int(entry["state"]["floor"]) > LAST_PRE_B3A:
                continue  # the chain adds nothing past floor 47 (after the first Act 3 boss): not a screen it can change
            view = shards.state_view(rec, entry)
            a_, b_ = champ.pick(view, decision, None), cand.pick(view, decision, None)
            cell["eligible"] += 1
            cell["changed"] += int(a_.index != b_.index)
            cell["champion_agrees"] += int(a_.index == int(entry["chosen"]))
            if act == 2 and entry["type"] == "rest" and view.max_hp > 0 and view.hp * 100 < 70 * view.max_hp and any(o.kind == "rest.heal" for o in options):
                side["rest_low_hp"]["screens"] += 1
                side["rest_low_hp"]["champion_heals"] += int(options[a_.index].kind == "rest.heal")
                side["rest_low_hp"]["candidate_heals"] += int(options[b_.index].kind == "rest.heal")
            if act == 2 and entry["type"] == "shop":
                for who, pk in (("champion", a_), ("candidate", b_)):
                    kind_ = options[pk.index].kind
                    if kind_ != "shop.leave":
                        side["shop_act3"][f"purchases_{who}"] += 1
                        side["shop_act3"][f"potions_{who}"] += int(kind_ in ("shop.potion", "shop.potion_swap"))
            if act in (1, 2) and entry["type"] == "map":
                side["route_act23"]["screens"] += 1
                for who, pk in (("champion", a_), ("candidate", b_)):
                    node = options[pk.index].node
                    side["route_act23"][f"{who}_elite"] += int(node is not None and int(node[2]) == vocab.NODE_ELITE)
        done += 1
    acts: dict[str, Any] = {}
    for key, c in sorted(by.items()):
        a = key.split("/")[0]
        t = acts.setdefault(a, _pick_stats())
        for k in t:
            t[k] += c[k]

    def share(num: int, den: int) -> float | None:
        return num / den if den >= MIN_CELL else None

    for c in [*by.values(), *acts.values()]:
        c["changed_share"] = share(c["changed"], c["eligible"])
        c["champion_agreement"] = share(c["champion_agrees"], c["eligible"])
    rl, sh, ro = side["rest_low_hp"], side["shop_act3"], side["route_act23"]
    side_out = {"rest_low_hp": {**rl, "champion_heal_share": share(rl["champion_heals"], rl["screens"]), "candidate_heal_share": share(rl["candidate_heals"], rl["screens"])},
                "shop_act3": {**sh, "champion_potion_share": share(sh["potions_champion"], sh["purchases_champion"]),
                              "candidate_potion_share": share(sh["potions_candidate"], sh["purchases_candidate"])},
                "route_act23": {**ro, "champion_elite_share": share(ro["champion_elite"], ro["screens"]), "candidate_elite_share": share(ro["candidate_elite"], ro["screens"])}}
    tot = _pick_stats()
    for t in acts.values():
        for k in tot:
            tot[k] += t[k]
    tot["champion_agreement"] = share(tot["champion_agrees"], tot["eligible"])
    return {"written": utc_now(), "runs_replayed": done, "by_act": acts, "by_screen": by, "side": side_out, "all": tot,
            "champion": Path(champion_dir).name, "candidate": Path(candidate_dir).name}


def flip_vetoes(doc: dict[str, Any]) -> list[str]:
    """The pre-registered vetoes on a flips document (brief step 10): an Act 1 decision changed, the champion's replay agreement under 98 in 100, an Act 2 or
    Act 3 changed share over 50 in 100, the Act 3 rest sites' heal share (HP under 70 in 100) falling by more than 0.10."""
    out = []
    a1 = doc["by_act"].get("act1")
    if a1 and a1["changed"] > 0:
        out.append(f"{a1['changed']} Act 1 decisions changed")
    agree = doc["all"].get("champion_agreement")
    if agree is not None and agree < 0.98:
        out.append(f"the champion's replay agreement is {agree:.3f}, under 0.98")
    for act in ("act2", "act3"):
        c = doc["by_act"].get(act)
        if c and c.get("changed_share") is not None and c["changed_share"] > 0.50:
            out.append(f"{act} changed share {c['changed_share']:.3f} is over 0.50")
    r = doc["side"]["rest_low_hp"]
    if r["champion_heal_share"] is not None and r["candidate_heal_share"] is not None and r["champion_heal_share"] - r["candidate_heal_share"] > 0.10:
        out.append(f"Act 3 low-HP heal share falls {r['champion_heal_share']:.3f} -> {r['candidate_heal_share']:.3f}")
    return out


def cmd_flips(a: argparse.Namespace) -> int:
    below_normal()
    doc = flips(a.champion, a.candidate, a.store, a.runs)
    doc["vetoes"] = flip_vetoes(doc)
    a3 = doc["by_act"].get("act3", {})
    doc["expect_no_effect"] = bool(a3.get("changed_share") is not None and a3["changed_share"] < 0.03)
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(doc, indent=1, default=float), encoding="utf-8")
    print(json.dumps({"runs": doc["runs_replayed"], "by_act": {k: {kk: v[kk] for kk in ("eligible", "changed", "changed_share", "champion_agreement")} for k, v in doc["by_act"].items()},
                      "side": doc["side"], "vetoes": doc["vetoes"], "expect_no_effect": doc["expect_no_effect"]}, default=float))
    return 0


# ------------------------------------------------------------------------------------------------------------------ cli


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("build")
    b.add_argument("--stores", default=",".join(map(str, STORES)))
    b.add_argument("--run-value", default=str(RUN_VALUE))
    b.add_argument("--out", default=str(LOGS / "rows"))
    b.add_argument("--procs", type=int, default=4)
    b.add_argument("--store-root", default=str(STORE_ROOT))
    b.add_argument("--max-shards", type=int, default=0, help="a stride sample of the shards (0 = all)")
    f = sub.add_parser("fit")
    f.add_argument("--rows", default=str(LOGS / "rows"))
    f.add_argument("--init", default=str(RUN_VALUE))
    f.add_argument("--out", default="D:/spireline/referee/agent/boss-chain/v1")
    f.add_argument("--seeds", default="0,1,2")
    f.add_argument("--epochs", type=int, default=4)
    f.add_argument("--batch", type=int, default=512)
    f.add_argument("--lr", type=float, default=1e-4)
    f.add_argument("--weight-decay", type=float, default=1e-2)
    f.add_argument("--dropout", type=float, default=0.3)
    f.add_argument("--evals-per-epoch", type=int, default=4)
    f.add_argument("--patience", type=int, default=8)
    f.add_argument("--device", default=None)
    r = sub.add_parser("report")
    r.add_argument("--rows", default=str(LOGS / "rows"))
    r.add_argument("--model", default="D:/spireline/referee/agent/boss-chain/v1/boss_chain.npz")
    r.add_argument("--out", default="D:/spireline/referee/agent/boss-chain/v1")
    r.add_argument("--champion", action="store_true")
    fl = sub.add_parser("flips")
    fl.add_argument("--champion", required=True)
    fl.add_argument("--candidate", required=True)
    fl.add_argument("--store", default=str(STORE_ROOT / "18"))
    fl.add_argument("--runs", type=int, default=400)
    fl.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    return {"build": cmd_build, "fit": cmd_fit, "report": cmd_report, "flips": cmd_flips}[a.cmd](a)


if __name__ == "__main__":
    sys.exit(main())
