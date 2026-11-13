"""The seed-matched run exam (D58 addendum 14): our player on the seeds of human runs, read against the humans' own outcomes on those seeds.

    python rl/run_seed_exam.py play   --fifth 4 --seeds 400 --draws 2 --agent agent/current.json --out DIR [--procs 4] [--config a3-v1]
    python rl/run_seed_exam.py report --out DIR [--write]

A human run's seed regenerates its maps, its first-room gift, its shops and its card offers on the copy (`docs/diagnostics/2026-11-05-human-seed-replay.md`).
`play` picks `--seeds` human runs of one training fifth (half won by the human, half lost, by a hash of the seed), plays each seed with the current
player `--draws` times (the lab's own run loop, `gen0.play_emu_run`: the same engine, Ascension 10, Ironclad, the agent's search schedule at the lab's
config; one emulator per process, a spawn pool, BelowNormal), and writes the player's per-seed outcome rows next to the humans' aggregates on the same
seeds. A draw is the same seed with another stream of the player's own search, so the fights come out differently once a decision differs; that is the
point. `report` prints the table: win rate, floor reached, the Act 1 and Act 2 boss reached and killed, HP at the door, for the player and for the
humans on the same seeds, all and split by whether the human won the run, beside the copy's figures on the lab's own seeds ("are these seeds easier").

What leaves the corpus (D13, D38, D55): aggregates, and our own per-seed rows keyed by `seed_key` (a salted hash of the seed). A seed is a derived
input: it is read, held in memory and handed to a worker, never written to a result, a log or an error text (`scrub`). No run id and no player id is
written (the player id is never read). A cell over fewer than 20 runs is hidden. The exam fifth (hash bucket 0) is refused unless `--exam` is given
with `--milestone NAME`, and each use of it appends one line to the use log; the first reading never touches it.

Source of the human figures: Spire Codex public run export for v0.107.1. Module level imports are the standard library and `run_human_bar` (itself
standard library only); the engine and the agent are imported inside `play`.
"""

from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import math
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import run_human_bar as hb  # noqa: E402
from sts2_gym import faults  # noqa: E402 -- pure: loads no native code

SPIRE_CODEX = hb.SPIRE_CODEX
MIN_CELL = hb.MIN_CELL  # D13: a cell over fewer runs is not shown
VERSION = 1
SALT = "spireline-seed-exam|"
KEY_LEN = 16  # hex characters of a seed's key
TRAIN_FIFTHS = (1, 2, 3, 4)
EXAM_FIFTH = 0
REFEREE = Path("D:/spireline/referee")
OUT_ROOT = REFEREE / "seed-exam"
EXAM_USE_LOG = OUT_ROOT / "exam-use.log"
DEFAULT_AGENT = REFEREE / "agent" / "current.json"
DEFAULT_CONFIG = "a3-v1"
ASCENSION = 10
MAX_DRAWS = 64
ACTS = (1, 2)
Z = 1.959964
RECORDS_ENV = "SPIRELINE_EXAM_RECORDS"  # the folder `--records` names, handed to the spawn workers through the environment
FINISHED = ("win", "death")  # the rows whose run ran to its end; a crash, a timeout or a lost worker is counted and left out of the means


# ---------------------------------------------------------------------------------------------------------------------- the seed's key


def seed_key(seed: str) -> str:
    """The row key of a seed: a salted hash. The order of the pick is this hash too (the smallest first), so the pick never depends on the order of the corpus."""
    return hashlib.sha256((SALT + str(seed)).encode()).hexdigest()[:KEY_LEN]


def draw_index(key: str, draw: int) -> int:
    """The run index the lab's agent streams are seeded with (`default_rng([agent seed, ix])`): one per (seed, draw), a function of the key alone."""
    return (int(key, 16) % (1 << 40)) * MAX_DRAWS + int(draw)


def scrub(text: Any, seeds: Iterable[str], key_of: Callable[[str], str] = seed_key) -> str:
    """`text` with every seed in `seeds` replaced by its key (an error message from the engine may name the seed it was given)."""
    out = str(text)
    for s in seeds:
        if s and s in out:
            out = out.replace(s, key_of(s))
    return out


# ---------------------------------------------------------------------------------------------------------------------- which fifth


def check_fifth(fifth: int | None, exam: bool, milestone: str | None) -> int:
    """The fifth to read. The exam fifth needs `--exam` with a named milestone (and `--exam` means that fifth and no other); a training fifth is 1 to 4."""
    if exam:
        if not milestone or not str(milestone).strip():
            raise SystemExit("run_seed_exam: --exam reads the exam fifth and needs --milestone NAME (the use is written to the use log)")
        if fifth not in (EXAM_FIFTH, None):
            raise SystemExit(f"run_seed_exam: --exam is the exam fifth ({EXAM_FIFTH}); --fifth {fifth} is a training fifth")
        return EXAM_FIFTH
    fifth = 4 if fifth is None else fifth
    if fifth == EXAM_FIFTH:
        raise SystemExit("run_seed_exam: fifth 0 is the exam fifth; it is read only with --exam --milestone NAME")
    if fifth not in TRAIN_FIFTHS:
        raise SystemExit(f"run_seed_exam: --fifth must be one of {TRAIN_FIFTHS}, not {fifth}")
    return int(fifth)


def record_exam_use(path: Path | str, milestone: str, seeds: int, draws: int, agent: str) -> str:
    """One line of the use log for one use of the exam fifth: when, which milestone, how many seeds and draws, which agent. No seed, no id."""
    line = f"{dt.datetime.now(dt.UTC).strftime('%Y-%m-%dT%H:%M:%SZ')} exam-fifth milestone={str(milestone).strip()} seeds={seeds} draws={draws} agent={agent}"
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "a", encoding="utf-8") as fh:
        fh.write(line + "\n")
    return line


# ---------------------------------------------------------------------------------------------------------------------- the human runs


def read_fifth(corpus: Path | str, fifth: int, exam_ids: frozenset[str], stats: Counter, limit: int = 0) -> Iterator[tuple[str, dict[str, Any]]]:
    """`(seed, human_run)` for every usable run (Ironclad, Ascension 10, standard, finished, one player, no modifiers, vanilla acts) of hash bucket
    `fifth`. A training fifth is read through `human_prior.iter_run_lines`, which drops the exam fifth before any record is seen, and `assert_no_exam`
    stands behind it. The exam fifth (fifth 0) is read by this function's own filter, only when `check_fifth` let it through. The run id is used for the
    bucket and dropped; the player id is never read."""
    import human_prior as hp  # noqa: PLC0415
    import human_starts as hs  # noqa: PLC0415

    if fifth == EXAM_FIFTH:
        lines = _exam_lines(corpus, exam_ids, stats, hs)
    else:
        lines = hp.iter_run_lines(Path(corpus), exam_ids, stats)
    n = 0
    for line in lines:
        r = json.loads(line)
        rid = hs.run_id(r)
        if hs.exam_bucket(rid) != fifth:
            stats["other_fifth_left_out"] += 1
            continue
        if fifth != EXAM_FIFTH:
            hs.assert_no_exam([rid], exam_ids)
        seed = r.get("seed")
        row = hb.human_run(r)
        if not isinstance(seed, str) or not seed:
            stats["no_seed"] += 1
            continue
        if row is None:
            stats["no_map_history"] += 1
            continue
        yield seed, row
        n += 1
        if limit and n >= limit:
            return


def _exam_lines(corpus: Path | str, exam_ids: frozenset[str], stats: Counter, hs: Any) -> Iterator[str]:
    """The raw lines of the exam fifth's usable runs (the same filters as `human_prior.iter_run_lines`, which drops them)."""
    import gzip  # noqa: PLC0415

    opener = gzip.open if str(corpus).endswith(".gz") else open
    with opener(corpus, "rt", encoding="utf-8") as fh:
        for line in fh:
            stats["lines"] += 1
            if '"ascension":10,' not in line or '"character":"CHARACTER.IRONCLAD"' not in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                stats["unparsable"] += 1
                continue
            if not hs.human_ok(r) or not hs.is_exam(hs.run_id(r), exam_ids):
                continue
            if not hs.vanilla_acts(r):
                stats["modded_acts"] += 1
                continue
            stats["runs_usable"] += 1
            yield line


def pick(entries: Iterable[tuple[str, Mapping[str, Any]]], n: int) -> tuple[list[dict[str, Any]], dict[str, int]]:
    """`n` runs: half the ones the human won and half the ones it lost (the odd one to the lost), each class by the smallest seed key first, one run
    per seed. Returns the picks `{key, seed, row, won}` (won first, each by key) and the counts: `available` by class, `picked` by class, `duplicate_seed`."""
    seen: set[str] = set()
    pool: dict[bool, list[dict[str, Any]]] = {True: [], False: []}
    dup = 0
    for seed, row in entries:
        key = seed_key(seed)
        if key in seen:
            dup += 1
            continue
        seen.add(key)
        pool[bool(row.get("win"))].append({"key": key, "seed": seed, "row": dict(row), "won": bool(row.get("win"))})
    want = {True: n // 2, False: n - n // 2}
    chosen: list[dict[str, Any]] = []
    picked = {}
    for won in (True, False):
        ranked = sorted(pool[won], key=lambda e: e["key"])[: want[won]]
        picked["won" if won else "lost"] = len(ranked)
        chosen.extend(ranked)
    return chosen, {"available_won": len(pool[True]), "available_lost": len(pool[False]), "picked_won": picked["won"], "picked_lost": picked["lost"],
                    "duplicate_seed": dup}


# ---------------------------------------------------------------------------------------------------------------------- the figures


def mean_ci(values: Sequence[float]) -> dict[str, Any] | None:
    """Mean and the 95% interval of the mean over independent units (seeds); None under MIN_CELL units."""
    n = len(values)
    if n < MIN_CELL:
        return None
    m = sum(values) / n
    sd = math.sqrt(sum((v - m) ** 2 for v in values) / (n - 1))
    h = Z * sd / math.sqrt(n)
    return {"n": n, "mean": round(m, 4), "lo": round(m - h, 4), "hi": round(m + h, 4)}


def ours_run(row: Mapping[str, Any]) -> dict[str, Any] | None:
    """One finished row (ours, or a lab arm's) as the run-level figures of `run_human_bar.summarise`: `floor`, `win`, and for each of Acts 1 and 2 whose
    boss it reached `{hp, max_hp, killed}`. A row of a run that did not run to its end (crash, timeout, lost) is None."""
    end = row.get("end")
    if not isinstance(row.get("floor"), int) or row.get("crash") or row.get("lost") or (end is not None and end not in FINISHED):
        return None
    door = row.get("door") if isinstance(row.get("door"), dict) else {}
    doors: dict[int, dict[str, Any]] = {}
    for ent in row.get("boss_damage_acts") or []:
        if isinstance(ent, (list, tuple)) and len(ent) == 2 and ent[0] in (0, 1):
            act = int(ent[0]) + 1
            hp = door.get(str(act)) or [None, None]
            doors[act] = {"hp": hp[0], "max_hp": hp[1], "killed": float(ent[1]) >= 1.0}
    return {"floor": int(row["floor"]), "win": bool(row.get("win")), "doors": doors}


def summarise_rows(rows: Iterable[Mapping[str, Any]]) -> dict[str, Any]:
    """`run_human_bar.summarise` of the finished rows, the same cells for ours and for the humans, with a seed-clustered interval on the win rate and the
    mean floor when the rows carry a key (two draws of one seed are not two independent runs)."""
    rows = list(rows)
    runs = [r for r in (ours_run(x) for x in rows) if r is not None]
    out = hb.summarise(runs)
    out["excluded"] = len(rows) - len(runs)
    by_seed: dict[str, list[Mapping[str, Any]]] = {}
    for x, r in ((x, ours_run(x)) for x in rows):
        if r is not None and x.get("key"):
            by_seed.setdefault(x["key"], []).append(r)
    if by_seed:
        out["seeds"] = len(by_seed) if len(by_seed) >= MIN_CELL else None
        out["win_rate_ci"] = mean_ci([sum(1 for r in v if r["win"]) / len(v) for v in by_seed.values()])
        out["floor_ci"] = mean_ci([sum(r["floor"] for r in v) / len(v) for v in by_seed.values()])
    return out


def human_blocks(picks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """The humans' figures on the picked seeds: all, and split by whether the human won the run. Aggregates only."""
    return {"all": hb.summarise([p["row"] for p in picks]), "won": hb.summarise([p["row"] for p in picks if p["won"]]),
            "lost": hb.summarise([p["row"] for p in picks if not p["won"]])}


def ours_blocks(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    return {"all": summarise_rows(rows), "won": summarise_rows([r for r in rows if r.get("human_won")]),
            "lost": summarise_rows([r for r in rows if not r.get("human_won")])}


def paired_blocks(picks: Sequence[Mapping[str, Any]], rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Our player against the human on the SAME seed, as aggregates (the human's per-seed figures stay in memory): the floor difference, the share of
    seeds where our best draw got at least as far, how many human-won seeds some draw of ours won, and per act the share of our draws that killed the boss
    on the seeds where the human killed it. Under MIN_CELL seeds a cell is None."""
    done: dict[str, list[dict[str, Any]]] = {}
    for x in rows:
        r = ours_run(x)
        if r is not None and x.get("key"):
            done.setdefault(x["key"], []).append(r)
    out: dict[str, Any] = {}
    for label, want in (("all", None), ("won", True), ("lost", False)):
        seeds = [p for p in picks if p["key"] in done and (want is None or p["won"] == want)]
        cell: dict[str, Any] = {"seeds": len(seeds) if len(seeds) >= MIN_CELL else None}
        if len(seeds) >= MIN_CELL:
            diffs = [sum(r["floor"] for r in done[p["key"]]) / len(done[p["key"]]) - p["row"]["floor"] for p in seeds]
            cell["floor_diff"] = mean_ci(diffs)
            cell["best_draw_at_least_human_floor"] = round(sum(1 for p in seeds if max(r["floor"] for r in done[p["key"]]) >= p["row"]["floor"]) / len(seeds), 4)
            cell["some_draw_won"] = sum(1 for p in seeds if any(r["win"] for r in done[p["key"]]))
            for act in ACTS:
                both = [p for p in seeds if act in p["row"]["doors"] and p["row"]["doors"][act]["killed"]]
                if len(both) < MIN_CELL:
                    cell[f"act{act}_boss_killed_where_human_did"] = None
                    continue
                share = [sum(1 for r in done[p["key"]] if act in r["doors"] and r["doors"][act]["killed"]) / len(done[p["key"]]) for p in both]
                cell[f"act{act}_boss_killed_where_human_did"] = {"seeds": len(both), "share": round(sum(share) / len(share), 4)}
        out[label] = cell
    return out


def own_seed_rows(agent_path: Path | str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """The copy's own-seed figures: the rows of the judgement arms the lab played with this same agent (`<agent dir>/arms/*/runs.jsonl` whose summary
    names the agent's generation, folder, stakes schedule and macro). Returns (rows, description). No arms found: ([], {})."""
    try:
        cur = json.loads(Path(agent_path).read_text(encoding="utf-8"))
        d = Path(cur["dir"]) if "dir" in cur else Path(agent_path).parent
        meta = json.loads((d / "agent.json").read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError):
        return [], {}
    tag = (meta.get("search_schedule") or {}).get("tag")
    rows: list[dict[str, Any]] = []
    used = 0
    for arm in sorted((d / "arms").glob("*/runs.jsonl")) if (d / "arms").is_dir() else []:
        try:
            summ = json.loads((arm.parent / "summary.json").read_text(encoding="utf-8"))
        except (OSError, ValueError):
            continue
        named = summ.get("agent") or {}
        if named.get("generation") != meta.get("generation") or named.get("macro") != meta.get("macro") or named.get("stakes") != tag:
            continue
        used += 1
        for line in arm.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if isinstance(r, dict):
                rows.append({k: r.get(k) for k in ("floor", "win", "end", "crash", "lost", "boss_damage_acts")})
    return rows, {"generation": meta.get("generation"), "arms": used, "config": DEFAULT_CONFIG}


# ---------------------------------------------------------------------------------------------------------------------- the table


def _f(x: Any, fmt: str = "{:.1f}") -> str:
    return hb._f(x, fmt)


def _p(x: Any) -> str:
    return hb._pct(x)


def _col(block: Mapping[str, Any] | None) -> dict[str, str]:
    """The cells of one column of the table from a `summarise`-shaped block (a hidden cell is '-')."""
    b = block or {}
    fl = b.get("floor") or {}
    cell = {"runs": "-" if b.get("runs") is None else str(b["runs"]), "win": _p(b.get("win_rate")), "floor": _f(fl.get("mean")),
            "median": _f(fl.get("p50"))}
    ci = b.get("win_rate_ci")
    if ci:
        cell["win"] += f" [{100 * ci['lo']:.1f}, {100 * ci['hi']:.1f}]"
    fci = b.get("floor_ci")
    if fci:
        cell["floor"] += f" [{fci['lo']:.1f}, {fci['hi']:.1f}]"
    for a in ACTS:
        ac = (b.get("acts") or {}).get(str(a)) or {}
        d = ac.get("door_hp") or {}
        cell[f"reach{a}"] = _p(ac.get("reached_share"))
        cell[f"kill{a}"] = _p(ac.get("kill_rate"))
        cell[f"hp{a}"] = "-" if d.get("mean") is None else f"{d['mean']:.1f} ({_p(d.get('share_of_max'))} of max)"
    return cell


def weighted_cells(won: Mapping[str, Any] | None, lost: Mapping[str, Any] | None, w: float) -> dict[str, Any] | None:
    """The figures of the two splits mixed at weight `w` on the seeds the human won (the fifth's own share of won runs, which the 200 + 200 pick is not):
    win rate, mean floor, and per act the boss reached share and the kill rate of those reaching it (a ratio of the mixed killed and reached shares).
    None when either split is hidden; an act's cells are None when either split's act cell is."""
    a, b = won or {}, lost or {}
    if a.get("runs") is None or b.get("runs") is None:
        return None
    out: dict[str, Any] = {"win": w * a["win_rate"] + (1 - w) * b["win_rate"], "floor": w * a["floor"]["mean"] + (1 - w) * b["floor"]["mean"]}
    for act in ACTS:
        aw, al = (a.get("acts") or {}).get(str(act)), (b.get("acts") or {}).get(str(act))
        if aw is None or al is None:
            out[f"reach{act}"] = out[f"kill{act}"] = None
            continue
        reach = w * aw["reached_share"] + (1 - w) * al["reached_share"]
        out[f"reach{act}"] = reach
        out[f"kill{act}"] = (w * aw["reached_share"] * aw["kill_rate"] + (1 - w) * al["reached_share"] * al["kill_rate"]) / reach if reach else None
    return out


def won_share(picked: Mapping[str, Any] | None) -> float | None:
    """The share of the fifth's usable runs the human won (the picks' `available` counts); None without them."""
    p = picked or {}
    n = (p.get("available_won") or 0) + (p.get("available_lost") or 0)
    return p["available_won"] / n if n else None


def table_text(doc: Mapping[str, Any]) -> str:
    """The exam's table from `summary.json`-shaped `doc`: the player and the humans on the same seeds (all, seeds the human won, seeds the human lost),
    the copy on its own seeds, then the paired figures. A cell under 20 runs shows '-'."""
    ours, hum, own = doc.get("ours") or {}, doc.get("human") or {}, doc.get("own_seeds") or {}
    cols = [("our player", ours.get("all")), ("humans, same seeds", hum.get("all")), ("our player, human won", ours.get("won")),
            ("humans, won", hum.get("won")), ("our player, human lost", ours.get("lost")), ("humans, lost", hum.get("lost")),
            (f"copy on its own seeds (gen {doc.get('own_arms', {}).get('generation', '?')})", own)]
    cells = [_col(b) for _, b in cols]
    rows = [("runs", "runs"), ("win rate", "win"), ("mean floor reached", "floor"), ("median floor", "median")]
    for a in ACTS:
        rows += [(f"Act {a} boss reached (of runs)", f"reach{a}"), (f"Act {a} boss killed (of those reaching it)", f"kill{a}"),
                 (f"Act {a} HP at the door", f"hp{a}")]
    ag = doc.get("agent") or {}
    lines = [f"seed-matched run exam ({SPIRE_CODEX}; Ironclad, Ascension 10; fifth {doc.get('fifth')}; {doc.get('seeds')} seeds x {doc.get('draws')} draws; "
             f"agent generation {ag.get('generation')}, config {doc.get('config')}, engine {doc.get('engine')})", "",
             "| figure | " + " | ".join(n for n, _ in cols) + " |", "|---|" + "---|" * len(cols)]
    for name, k in rows:
        lines.append(f"| {name} | " + " | ".join(c[k] for c in cells) + " |")
    lines += ["", "Ours: runs that ran to their end; brackets are 95% intervals over seeds (two draws of a seed are not independent). The humans' columns are the "
              "records of the same seeds, so by construction their win rate is 100 in 100 on the seeds the human won and 0 on the ones it lost; the "
              "'all' columns are half and half. The copy's own seeds are the lab's judgement blocks of the same agent (no HP at the door there)."]
    w = won_share(doc.get("picked"))
    if w is not None:
        wo, wh = weighted_cells(ours.get("won"), ours.get("lost"), w), weighted_cells(hum.get("won"), hum.get("lost"), w)
        lines += ["", f"Weighted to the fifth's own mix ({100 * w:.1f} in 100 of its usable runs were won by the human; the pick is half and half), "
                      "the two splits of each side mixed at that share:", "", "| figure | our player | humans |", "|---|---|---|"]
        for name, k, fmt in (("win rate", "win", _p), ("mean floor reached", "floor", _f), ("Act 1 boss reached (of runs)", "reach1", _p),
                             ("Act 1 boss killed (of those reaching it)", "kill1", _p), ("Act 2 boss reached (of runs)", "reach2", _p),
                             ("Act 2 boss killed (of those reaching it)", "kill2", _p)):
            lines.append(f"| {name} | {fmt((wo or {}).get(k))} | {fmt((wh or {}).get(k))} |")
    paired = doc.get("paired") or {}
    if paired:
        lines += ["", "On the same seed (our draws against the human's run, aggregates):", "",
                  "| figure | all seeds | human won | human lost |", "|---|---|---|---|"]

        def pc(f: Callable[[Mapping[str, Any]], str]) -> str:
            return " | ".join(f(paired.get(k) or {}) for k in ("all", "won", "lost"))

        def fd(c: Mapping[str, Any]) -> str:
            d = c.get("floor_diff")
            return "-" if not d else f"{d['mean']:+.1f} [{d['lo']:+.1f}, {d['hi']:+.1f}]"

        lines.append("| seeds | " + pc(lambda c: "-" if c.get("seeds") is None else str(c["seeds"])) + " |")
        lines.append("| floor, ours minus the human's | " + pc(fd) + " |")
        lines.append("| our best draw reached at least the human's floor | " + pc(lambda c: _p(c.get("best_draw_at_least_human_floor"))) + " |")
        lines.append("| seeds where some draw of ours won | " + pc(lambda c: "-" if c.get("some_draw_won") is None else str(c["some_draw_won"])) + " |")
        for a in ACTS:
            lines.append(f"| Act {a} boss: our draws that killed it, where the human did | "
                         + pc(lambda c, a=a: "-" if not c.get(f"act{a}_boss_killed_where_human_did") else
                              f"{_p(c[f'act{a}_boss_killed_where_human_did']['share'])} of {c[f'act{a}_boss_killed_where_human_did']['seeds']} seeds") + " |")
    lines += ["", "A cell over fewer than 20 runs (or 20 seeds) is not shown (-)."]
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------------------------------------------------- files


def write_json(path: Path, doc: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_rows(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if path.is_file():
        for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
            try:
                r = json.loads(line)
            except ValueError:
                continue  # a line cut by a stop
            if isinstance(r, dict) and r.get("key") is not None:
                rows.append(r)
    return rows


def keep_resumable(out: Path, identity: dict[str, Any], force: bool) -> set[tuple[str, int]]:
    """The (key, draw) pairs a rerun into `out` keeps. The folder's identity (agent, config, engine, fifth, the picked keys, draws) must equal this run's;
    another identity is refused unless `force`, which starts the folder empty. The rows file is rewritten with the first row of each pair."""
    out.mkdir(parents=True, exist_ok=True)
    runs, stamp = out / "runs.jsonl", out / "resume.json"
    have = None
    if stamp.is_file():
        try:
            have = json.loads(stamp.read_text(encoding="utf-8"))
        except ValueError:
            have = None
    rows = read_rows(runs)
    if rows and have != identity:
        if not force:
            raise SystemExit("run_seed_exam: the folder holds rows of another exam (agent, setting or picks differ); use another --out or --force")
        rows = []
    kept, seen = [], set()
    for r in rows:
        k = (r["key"], int(r.get("draw", 0)))
        if k not in seen:
            seen.add(k)
            kept.append(r)
    tmp = runs.with_name("runs.jsonl.tmp")
    tmp.write_text("".join(json.dumps(r) + "\n" for r in kept), encoding="utf-8")
    os.replace(tmp, runs)
    write_json(stamp, identity)
    return seen


# ---------------------------------------------------------------------------------------------------------------------- one run (in a worker)


def _below_normal() -> None:
    hb._below_normal()


def _play_recorded(gen0: Any, seed: str, ix: int, config: str, ascension: int, agent: Any, hook: Callable[..., bool], records: str, key: str, draw: int) -> dict[str, Any]:
    """The same run with the lab's recording path on (`play_emu_run(record=True)`) and the measurement accounting of `seed_exam_records` on top of the
    recorder. The run's record becomes one derived file in `records` (the seed's key in its name, no seed inside); the row loses the combat samples."""
    import gen_loop  # noqa: PLC0415
    import seed_exam_records as ser  # noqa: PLC0415

    saved = gen_loop.RunRecorder
    gen_loop.RunRecorder = ser.recorder_class()
    try:
        row = gen0.play_emu_run(seed, ix, config, ascension, agent=agent, hook=hook, record=True)
    finally:
        gen_loop.RunRecorder = saved
    shard = row.pop("_shard", None)
    if shard is not None:
        ser.write_record(records, key, draw, row, shard["record"])
    return row


def play_task(seed: str, key: str, draw: int, ix: int, config: str, ascension: int, agent: Any, human_won: bool) -> dict[str, Any]:
    """One run of one seed under the agent (a worker's task): the lab's `gen0.play_emu_run` with a hook that reads the HP the first decision of each boss
    fight begins with (the door). The hook returns False, so the run is the lab's run unchanged. The row carries the seed's key, never the seed."""
    import gen0  # noqa: PLC0415

    gen0._setup()
    t0, c0 = time.perf_counter(), time.process_time()
    doors: dict[str, list[int]] = {}
    seen: set[int] = set()

    def hook(_k: int, _root: Any, s: Any, _pos: Any) -> bool:
        info = s.info
        floor = int(info["floor"])
        if info["node_type"] == hb.BOSS_NODE and floor not in seen:
            seen.add(floor)
            doors[str(int(s.act_index) + 1)] = [int(info["hp"]), int(info["max_hp"])]
        return False

    records = os.environ.get(RECORDS_ENV)
    try:
        if records:
            row = _play_recorded(gen0, seed, ix, config, ascension, agent, hook, records, key, draw)
        else:
            row = gen0.play_emu_run(seed, ix, config, ascension, agent=agent, hook=hook)
    except Exception as e:  # noqa: BLE001 -- an engine or agent defect is a crash of that run, counted and left out of the means
        row = {"win": False, "floor": 0, "end": "crash", "decisions": 0, "crash": True, "error": scrub(f"{type(e).__name__}: {e}", [seed], seed_key)[:300]}
        if (fault := faults.fault_of(e)) is not None:
            row.update(end="engine_fault", **fault.row())  # the engine's step threw (docs/ENGINE_FAULTS.md): counted by class
    row = {k: v for k, v in row.items() if k not in ("seed", "ix", "prior_calls", "prior_latency_ms", "_shard")}
    row.update(key=key, draw=draw, human_won=bool(human_won), door=doors, wall=round(time.perf_counter() - t0, 2), cpu=round(time.process_time() - c0, 2))
    return row


def worker_init() -> None:
    import gen0  # noqa: PLC0415

    gen0._init_worker()


# ---------------------------------------------------------------------------------------------------------------------- the driver


def run_block(picks: Sequence[Mapping[str, Any]], draws: int, *, procs: int, out: Path, config: str, ascension: int, agent: Any,
              task: Callable[..., dict[str, Any]] | None = None, identity: dict[str, Any] | None = None, force: bool = False,
              run_timeout_s: float | None = None, log: Callable[[str], None] = lambda s: print(s, flush=True)) -> dict[str, Any]:
    """Play every (seed, draw) of `picks` not already in `out/runs.jsonl`, draw by draw (so a stop leaves every seed with a first draw), on a spawn pool of
    `procs` single-emulator processes (`spawn_pool.SupervisedPool`); `procs` 0 plays in this process (tests). A worker that died is run again once, a
    timeout is written as a lost row at once. Every finished run is appended to `runs.jsonl`. Nothing written or logged names a seed."""
    task = task or play_task  # looked up now, so a test can replace it
    identity = identity or {}
    done = keep_resumable(out, identity, force)
    todo = [(p, d) for d in range(draws) for p in picks if (p["key"], d) not in done]
    seeds = [p["seed"] for p in picks]
    total = len(picks) * draws
    t0 = time.time()
    log(f"seed-exam: {len(done)} of {total} runs held, {len(todo)} to play on {procs} processes")
    counts = Counter()
    faults_by_class: Counter = Counter()  # the runs an engine fault ended, by class (docs/ENGINE_FAULTS.md)
    by_key = {p["key"]: p for p in picks}

    def lost(key: str, draw: int, why: str, wall: float) -> dict[str, Any]:
        return {"win": False, "floor": 0, "end": "lost", "lost": True, "key": key, "draw": draw, "human_won": by_key[key]["won"], "wall": round(wall, 2),
                "lost_reason": scrub(why, seeds)[:200]}

    with open(out / "runs.jsonl", "a", encoding="utf-8") as fh:

        def write(row: dict[str, Any]) -> None:
            fh.write(json.dumps(row) + "\n")
            fh.flush()
            counts["rows"] += 1
            counts[row.get("end", "?")] += 1
            for cls, n in faults.engine_fault_counts([row]).items():
                faults_by_class[cls] += n
            log(f"seed-exam: {len(done) + counts['rows']}/{total} key={row['key'][:8]} draw={row['draw']} end={row.get('end')} floor={row.get('floor')} "
                f"wall={row.get('wall')}s elapsed={round((time.time() - t0) / 60, 1)}min")

        args = lambda p, d: (p["seed"], p["key"], d, draw_index(p["key"], d), config, ascension, agent, p["won"])  # noqa: E731
        if procs <= 0:
            for p, d in todo:
                write(task(*args(p, d)))
        else:
            import multiprocessing as mp  # noqa: PLC0415

            import spawn_pool  # noqa: PLC0415

            pool = spawn_pool.SupervisedPool(mp.get_context("spawn"), procs, initializer=worker_init, recycle=20, run_timeout_s=run_timeout_s)
            attempts: dict[tuple[str, int], int] = {}
            retry: list[tuple[Mapping[str, Any], int]] = []
            queue = list(todo)
            try:
                while queue or retry or pool.inflight():
                    while pool.free() and (retry or queue):
                        p, d = retry.pop(0) if retry else queue.pop(0)
                        pool.submit((p["key"], d), task, args(p, d))
                    for dn in pool.collect():
                        key, d = dn.key
                        if dn.status == "ok":
                            write(dn.value)
                            continue
                        attempts[(key, d)] = attempts.get((key, d), 0) + 1
                        log(f"seed-exam: key={key[:8]} draw={d} {dn.status} after {dn.wall_s:g}s ({scrub(dn.value, seeds)[:120]})")
                        if dn.status != "timeout" and attempts[(key, d)] < 2:
                            counts["retried"] += 1
                            retry.append((by_key[key], d))
                        else:
                            write(lost(key, d, f"{dn.status}: {dn.value}", dn.wall_s))
                    pool.wait(0.5)
            finally:
                pool.terminate()
    return {"played": counts["rows"], "wall_s": round(time.time() - t0, 1), "ends": {k: v for k, v in counts.items() if k not in ("rows", "retried")},
            "retried": counts["retried"], "engine_faults": dict(sorted(faults_by_class.items()))}


# ---------------------------------------------------------------------------------------------------------------------- summary


def build_summary(meta: Mapping[str, Any], picks: Sequence[Mapping[str, Any]] | None, rows: Sequence[Mapping[str, Any]], human: Mapping[str, Any] | None,
                  paired: Mapping[str, Any] | None, own_rows: Sequence[Mapping[str, Any]], own_info: Mapping[str, Any]) -> dict[str, Any]:
    doc = {**meta, "ours": ours_blocks(rows), "human": human or (human_blocks(picks) if picks else {}), "own_seeds": summarise_rows(own_rows) if own_rows else {},
           "own_arms": dict(own_info)}
    doc["paired"] = paired if paired is not None else (paired_blocks(picks, rows) if picks else {})
    ends = Counter(r.get("end", "?") for r in rows)
    doc["rows"] = {"total": len(rows), "ends": dict(sorted(ends.items()))}
    return doc


def cmd_play(a: argparse.Namespace) -> int:
    fifth = check_fifth(a.fifth, a.exam, a.milestone)
    if not 1 <= a.draws <= MAX_DRAWS:
        raise SystemExit(f"run_seed_exam: --draws must be 1 to {MAX_DRAWS}")
    hb._below_normal()
    import human_starts as hs  # noqa: PLC0415

    if a.records:
        os.environ[RECORDS_ENV] = str(Path(a.records).resolve())  # before the pool spawns: its workers inherit the environment
    corpus = Path(a.corpus) if a.corpus else hs.CORPUS
    exam_ids = hs.read_exam_ids(a.exam_ids) if a.exam_ids else hs.read_exam_ids()
    out = Path(a.out)
    stats: Counter = Counter()
    t0 = time.time()
    picks, counts = pick(read_fifth(corpus, fifth, exam_ids, stats, limit=a.limit_runs), a.seeds)
    print(f"seed-exam: read the corpus in {round(time.time() - t0)} s: {sum(counts[k] for k in ('available_won', 'available_lost'))} usable runs of fifth {fifth} "
          f"({counts['available_won']} won by the human, {counts['available_lost']} lost); picked {counts['picked_won']} won + {counts['picked_lost']} lost", flush=True)
    if not picks:
        raise SystemExit("run_seed_exam: no run picked")
    if counts["picked_won"] < a.seeds // 2 or counts["picked_lost"] < a.seeds - a.seeds // 2:
        print(f"seed-exam: warning: fewer runs than asked ({counts['picked_won']} won, {counts['picked_lost']} lost of {a.seeds // 2} and {a.seeds - a.seeds // 2})", flush=True)

    if a.lib:
        os.environ["STS2_LIB_PATH"] = a.lib
    import gen0  # noqa: PLC0415
    import gen_loop  # noqa: PLC0415

    gen0.select_engine(a.engine)  # before any library loads: the spawn workers inherit it
    gen0._setup()
    agent_path = resolve_agent(a.agent)
    agent = gen_loop.load_agent(agent_path)
    if a.exam:
        line = record_exam_use(a.exam_use_log, a.milestone, len(picks), a.draws, f"gen{agent.generation}")
        print(f"seed-exam: exam fifth in use: {line}", flush=True)
    keys = sorted(p["key"] for p in picks)
    identity = {"agent": agent.describe(), "config": a.config, "ascension": ASCENSION, "engine": a.engine, "fifth": fifth, "draws": a.draws,
                "picks": hashlib.sha256("".join(keys).encode()).hexdigest()[:16]}
    timeout = gen0.RUN_TIMEOUT_S * (agent.wall_factor() if hasattr(agent, "wall_factor") else 1.0)
    human = human_blocks(picks)
    meta = {"version": VERSION, "source": SPIRE_CODEX, "character": "Ironclad", "ascension": ASCENSION, "fifth": fifth, "exam": bool(a.exam),
            "seeds": len(picks), "draws": a.draws, "config": a.config, "engine": a.engine, "agent": agent.describe(), "picked": counts,
            "read": {k: int(stats[k]) for k in ("lines", "ironclad_a10", "runs_usable", "exam_fifth_left_out", "modded_acts", "no_seed", "no_map_history") if k in stats}}
    out.mkdir(parents=True, exist_ok=True)
    write_json(out / "human.json", {**meta, "human": human})
    summ = run_block(picks, a.draws, procs=a.procs, out=out, config=a.config, ascension=ASCENSION, agent=agent, identity=identity, force=a.force, run_timeout_s=timeout)
    rows = read_rows(out / "runs.jsonl")
    own_rows, own_info = own_seed_rows(agent_path)
    doc = build_summary({**meta, "block": summ, "written": dt.datetime.now(dt.UTC).strftime("%Y-%m-%dT%H:%M:%SZ")}, picks, rows, human, None, own_rows, own_info)
    write_json(out / "summary.json", doc)
    (out / "table.md").write_text(table_text(doc), encoding="utf-8")
    print(table_text(doc), flush=True)
    return 0


def resolve_agent(arg: str | Path) -> Path:
    p = Path(arg)
    if p.exists():
        return p.resolve()
    q = REFEREE / p
    if q.exists():
        return q
    raise SystemExit(f"run_seed_exam: no agent at {arg}")


def cmd_report(a: argparse.Namespace) -> int:
    out = Path(a.out)
    try:
        summ = json.loads((out / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        summ = None
    try:
        hum = json.loads((out / "human.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        hum = None
    base = summ or hum
    if not base:
        raise SystemExit("run_seed_exam: the folder holds neither summary.json nor human.json: nothing to report")
    rows = read_rows(out / "runs.jsonl")
    own_rows, own_info = own_seed_rows(resolve_agent(a.agent)) if a.agent else ([], {})
    if not own_rows and summ:
        own_info = summ.get("own_arms") or {}
    meta = {k: v for k, v in base.items() if k not in ("ours", "human", "own_seeds", "own_arms", "paired", "rows")}
    doc = build_summary(meta, None, rows, (hum or {}).get("human") or (summ or {}).get("human"), (summ or {}).get("paired"),
                        own_rows, own_info)
    if not own_rows and summ:
        doc["own_seeds"] = summ.get("own_seeds") or {}
    text = table_text(doc)
    print(text)
    if a.write:
        write_json(out / "summary.json", doc)
        (out / "table.md").write_text(text, encoding="utf-8")
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("play", help="play the seeds of human runs of one fifth with the current player")
    p.add_argument("--fifth", type=int, default=None, help="the training fifth (1 to 4; default 4); the exam fifth only with --exam")
    p.add_argument("--seeds", type=int, default=400, help="runs picked: half won by the human, half lost")
    p.add_argument("--draws", type=int, default=2, help="plays of each seed (the player's own streams differ)")
    p.add_argument("--agent", default=str(DEFAULT_AGENT), help="a current.json or a generation folder (default: the lab's current agent)")
    p.add_argument("--config", default=DEFAULT_CONFIG, help="the lab's search config (default a3-v1; the agent's own schedule applies on top)")
    p.add_argument("--out", required=True)
    p.add_argument("--procs", type=int, default=4, help="emulator processes (spawn pool, BelowNormal); 0 plays in this process")
    p.add_argument("--engine", choices=("old", "core2"), default="old", help="the emulator's engine; the lab's self-play runs old")
    p.add_argument("--lib", help="directory of the native library (STS2_LIB_PATH; default: this tree's out)")
    p.add_argument("--exam", action="store_true", help="the exam fifth; refused without --milestone")
    p.add_argument("--milestone", help="the named milestone an exam use is recorded under")
    p.add_argument("--exam-use-log", default=str(EXAM_USE_LOG))
    p.add_argument("--corpus", help="the run export (default: the lab's copy)")
    p.add_argument("--exam-ids", help="the exam fifth's ids file (default: beside the corpus)")
    p.add_argument("--limit-runs", type=int, default=0, help="stop reading after this many usable runs of the fifth (a trial)")
    p.add_argument("--force", action="store_true", help="start --out empty when it holds rows of another exam")
    p.add_argument("--records", help="a folder for one derived record a run (fights, turns, decks, relics, decisions; `rl/seed_exam_records.py`); the run is unchanged")
    p.set_defaults(fn=cmd_play)
    r = sub.add_parser("report", help="print the table of a finished --out")
    r.add_argument("--out", required=True)
    r.add_argument("--agent", default=None, help="re-read the copy's own-seed figures from this agent's judgement arms")
    r.add_argument("--write", action="store_true", help="rewrite summary.json and table.md")
    r.set_defaults(fn=cmd_report)
    return ap


def main(argv: list[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    return int(a.fn(a))


if __name__ == "__main__":
    raise SystemExit(main())
