#!/usr/bin/env python3
"""The human fight bar, read from the practice loop's own plays (D55 addendum 10): where does the player fall short of the human?

    python tools/human_bar.py [--root D:/spireline/referee/gym-loop] [--out-dir D:/spireline/referee/backlog/human-bar]
                              [--n-last 4] [--gap-hp 15] [--min-plays 1] [--boot 1000]

The corpus's per-floor record gives every rebuilt human fight start the human's own result (`starts/human-outcomes.json`,
`rl/human_starts.py outcomes`). The practice loop plays those starts (`starts/plays.jsonl`: start, game, the teacher's root value, the
outcome in the search's currency; the outcome's end HP and win from the buffer shard's fight record). This tool puts the two side by
side and only reads: it writes nothing under the loop's folder.

  * The human-bar table: for the human starts played so far, over the latest `n_last` plays of each start (a start is one line; the
    plays are the loop's x16 teacher with whatever checkpoint was newest): the player's survival and HP lost against the human's on the
    same start, by act and kind and by enemy, the paired difference (player minus human) with a 95% bootstrap interval over starts,
    and counts. A death costs the whole HP going in on both sides (the exam's rule). A play that has no outcome (capped, or a segment
    a stop cut: its value labels are dropped) counts for nothing.
  * The list of starts where the human survived and every one of the player's latest plays died (needs `--min-plays` plays).
  * The list of starts where the player lost `--gap-hp` HP or more above the human's loss (the mean over its latest plays).

`diagnosis.json` holds the start ids and the figures; `diagnosis.md` holds aggregates only (a cell over fewer than 20 runs is not
shown) and carries no start id. Source of the human results: Spire Codex public run export for v0.107.1 (the four training fifths).
"""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import practice_mix as pm  # noqa: E402

LOOP_ROOT = Path("D:/spireline/referee/gym-loop")
OUT_DIR = Path("D:/spireline/referee/backlog/human-bar")
MIN_CELL = 20  # D13: a cell over fewer runs is not shown in the aggregates
DEATH_VALUE = pm.DEATH_VALUE
RULE = "human-bar-v1"
BOOT_SEED = 20261108


# ------------------------------------------------------------------------------------------------------------------------- reading


def read_lite(starts_dir: Path) -> dict[str, dict]:
    """{id: lite record} of the human starts (act, kind, enemy, HP going in, run)."""
    try:
        rows = json.loads((Path(starts_dir) / "human-starts.lite.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    return {r["id"]: r for r in rows}


def read_outcomes(starts_dir: Path) -> dict[str, dict]:
    try:
        doc = json.loads((Path(starts_dir) / pm.HUMAN_OUTCOMES).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {}
    outs = doc.get("outcomes") if isinstance(doc, dict) else None
    return outs if isinstance(outs, dict) else {}


def shard_fights(path: Path) -> dict[tuple, dict]:
    """{(start, game): the fight's record} of one buffer shard (only the fight table is read; the tokens stay on disk)."""
    import numpy as np  # noqa: PLC0415

    try:
        with np.load(path) as z:
            metas = json.loads(str(z["fights_meta"]))
    except (OSError, ValueError, KeyError, EOFError):
        return {}
    return {(m["start"], int(m["j"])): m for m in metas}


def load_plays(root: Path) -> list[dict]:
    """The plays log's rows in play order (read only: the loop's own sync writes it), each with the shard's record of the fight:
    `won` and `end_hp` when the shard still exists, else `won` from the outcome's sign and no end HP."""
    root = Path(root)
    rows = [r for r in pm._iter_jsonl(root / "starts" / "plays.jsonl") if "start" in r]  # noqa: SLF001
    cache: dict[str, dict] = {}
    for r in rows:
        name = r.get("shard")
        if name and name not in cache:
            cache[name] = shard_fights(root / "buffer" / "shards" / name)
        m = (cache.get(name) or {}).get((r["start"], int(r["j"])))
        if m is not None:
            r["won"], r["end_hp"] = bool(m.get("won")), int(m.get("end_hp") or 0)
        elif r.get("played") is not None:
            r["won"], r["end_hp"] = r["played"] > DEATH_VALUE, None
    return rows


def latest_plays(rows: list[dict], n_last: int) -> dict[str, list[dict]]:
    """Each start's latest `n_last` plays that have an outcome, in play order."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        if r.get("played") is not None and "won" in r:
            by[r["start"]].append(r)
    return {s: v[-n_last:] for s, v in by.items()}


def start_figures(plays: dict[str, list[dict]], lite: dict[str, dict], outcomes: dict[str, dict]) -> list[dict]:
    """One line a human start the loop has played and the sidecar has a result for: the player's figures over its latest plays and the
    human's. HP lost: the HP going in less the end HP of a win, the whole HP going in for a death (the exam's rule)."""
    out = []
    for sid, ps in plays.items():
        rec, h = lite.get(sid), outcomes.get(sid)
        if rec is None or h is None or not sid.startswith("human:"):
            continue
        hp_in = int(rec["hp"])
        lost = [hp_in - p["end_hp"] if p["won"] else hp_in for p in ps if p["won"] is False or p.get("end_hp") is not None]
        vals = [float(p["played"]) for p in ps]
        out.append({"start": sid, "run": rec["run"], "act": rec["act"], "kind": rec["kind"], "enemy": rec["enemy"], "hp_in": hp_in,
                    "plays": len(ps), "p_surv": sum(1 for p in ps if p["won"]) / len(ps), "all_died": all(not p["won"] for p in ps),
                    "p_hp_lost": sum(lost) / len(lost) if lost else None, "p_value": sum(vals) / len(vals),
                    "h_surv": 1.0 if h.get("human_survived") else 0.0, "h_hp_lost": h.get("human_hp_lost"), "h_value": h.get("human_l0")})
    return sorted(out, key=lambda f: f["start"])


# ----------------------------------------------------------------------------------------------------------------------- the table


def _mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def _boot_ci(xs: list[float], boot: int) -> tuple[float | None, float | None]:
    """The 95% percentile interval of the mean of `xs` (resampled with replacement; fixed seed)."""
    if len(xs) < 2 or boot <= 0:
        return None, None
    rnd = random.Random(BOOT_SEED)
    n = len(xs)
    means = sorted(sum(rnd.choices(xs, k=n)) / n for _ in range(boot))
    return means[int(0.025 * (boot - 1))], means[int(math.ceil(0.975 * (boot - 1)))]


def group_row(label: str, figs: list[dict], boot: int) -> dict:
    """The line of one group of starts: counts, the two sides' means and the paired differences (player minus human) with intervals."""
    d_surv = [f["p_surv"] - f["h_surv"] for f in figs]
    d_hp = [f["p_hp_lost"] - f["h_hp_lost"] for f in figs if f["p_hp_lost"] is not None and f["h_hp_lost"] is not None]
    d_val = [f["h_value"] - f["p_value"] for f in figs if f["h_value"] is not None]
    lo_s, hi_s = _boot_ci(d_surv, boot)
    lo_h, hi_h = _boot_ci(d_hp, boot)
    return {"cell": label, "starts": len(figs), "runs": len({f["run"] for f in figs}), "plays": sum(f["plays"] for f in figs),
            "player_survival": _mean(f["p_surv"] for f in figs), "human_survival": _mean(f["h_surv"] for f in figs),
            "survival_diff": _mean(d_surv), "survival_diff_lo": lo_s, "survival_diff_hi": hi_s,
            "player_hp_lost": _mean(f["p_hp_lost"] for f in figs), "human_hp_lost": _mean(f["h_hp_lost"] for f in figs),
            "hp_lost_diff": _mean(d_hp), "hp_lost_diff_lo": lo_h, "hp_lost_diff_hi": hi_h, "hp_lost_starts": len(d_hp),
            "player_value": _mean(f["p_value"] for f in figs), "human_value": _mean(f["h_value"] for f in figs), "value_short": _mean(d_val)}


def tables(figs: list[dict], boot: int) -> dict:
    """The rows by act and kind, by enemy (act, kind and enemy) and for all starts."""
    by_ak: dict[str, list[dict]] = defaultdict(list)
    by_enemy: dict[str, list[dict]] = defaultdict(list)
    for f in figs:
        by_ak[f"act{f['act']}-{f['kind']}"].append(f)
        by_enemy[f"act{f['act']} {f['kind']} {f['enemy']}"].append(f)
    order = ("act1-hallway", "act1-elite", "act1-boss", "act2-hallway", "act2-elite", "act2-boss")
    return {"by_act_kind": [group_row(k, by_ak[k], boot) for k in sorted(by_ak, key=lambda k: order.index(k) if k in order else 99)],
            "by_enemy": [group_row(k, by_enemy[k], boot) for k in sorted(by_enemy)],
            "all": group_row("all", figs, boot) if figs else None}


# ------------------------------------------------------------------------------------------------------------------------ the lists


def short_lists(figs: list[dict], gap_hp: float, min_plays: int, n_last: int) -> dict:
    """The starts the human survived where every one of the player's latest plays died, and the starts where the player lost `gap_hp`
    HP or more above the human's loss (the mean over its latest plays; a death loses the whole HP going in)."""
    died = [f for f in figs if f["h_surv"] == 1.0 and f["all_died"] and f["plays"] >= min_plays]
    lost = [f for f in figs if f["plays"] >= min_plays and f["p_hp_lost"] is not None and f["h_hp_lost"] is not None
            and f["p_hp_lost"] - f["h_hp_lost"] >= gap_hp]

    def row(f):
        return {"start": f["start"], "act": f["act"], "kind": f["kind"], "enemy": f["enemy"], "hp_in": f["hp_in"], "plays": f["plays"],
                "player_survival": round(f["p_surv"], 3), "player_hp_lost": None if f["p_hp_lost"] is None else round(f["p_hp_lost"], 2),
                "human_survived": bool(f["h_surv"]), "human_hp_lost": f["h_hp_lost"],
                "player_value": round(f["p_value"], 2), "human_value": f["h_value"]}

    return {"human_survived_player_died": [row(f) for f in sorted(died, key=lambda f: (-f["plays"], f["start"]))],
            "player_lost_more": [row(f) for f in sorted(lost, key=lambda f: (-(f["p_hp_lost"] - f["h_hp_lost"]), f["start"]))],
            "n_last": n_last, "gap_hp": gap_hp, "min_plays": min_plays}


def by_enemy_counts(lst: list[dict], figs: list[dict]) -> list[dict]:
    """How many starts of each enemy are on a list, of the starts played (runs counted for the hiding rule)."""
    runs: dict[str, set] = defaultdict(set)
    played: Counter = Counter()
    for f in figs:
        k = f"act{f['act']} {f['kind']} {f['enemy']}"
        runs[k].add(f["run"])
        played[k] += 1
    on = Counter(f"act{r['act']} {r['kind']} {r['enemy']}" for r in lst)
    return [{"cell": k, "on_list": on[k], "played": played[k], "runs": len(runs[k])} for k in sorted(played)]


# ------------------------------------------------------------------------------------------------------------------------ the output


def pct(x: float | None) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def num(x: float | None, d: int = 1) -> str:
    return "-" if x is None else f"{x:.{d}f}"


def ci(v: float | None, lo: float | None, hi: float | None, scale: float = 1.0, d: int = 1) -> str:
    if v is None:
        return "-"
    rng = "" if lo is None or hi is None else f" [{scale * lo:+.{d}f}, {scale * hi:+.{d}f}]"
    return f"{scale * v:+.{d}f}{rng}"


def table_md(rows: list[dict], first: str) -> list[str]:
    out = [f"| {first} | starts | plays | player survival % | human % | diff pts [95%] | player HP lost | human HP lost | diff HP [95%] | value short |",
           "|---|---|---|---|---|---|---|---|---|---|"]
    hidden = 0
    for r in rows:
        if r["runs"] < MIN_CELL:
            hidden += 1
            continue
        out.append(f"| {r['cell']} | {r['starts']:,} | {r['plays']:,} | {pct(r['player_survival'])} | {pct(r['human_survival'])} | "
                   f"{ci(r['survival_diff'], r['survival_diff_lo'], r['survival_diff_hi'], 100)} | {num(r['player_hp_lost'])} | "
                   f"{num(r['human_hp_lost'])} | {ci(r['hp_lost_diff'], r['hp_lost_diff_lo'], r['hp_lost_diff_hi'])} | {num(r['value_short'])} |")
    if hidden:
        out += ["", f"{hidden} further cells are under {MIN_CELL} runs and not shown."]
    return out


def render_md(tab: dict, lists: dict, figs: list[dict], meta: dict) -> str:
    """The aggregates (no start id; a cell over fewer than MIN_CELL runs is not shown)."""
    out = ["# The human fight bar: where the player falls short of the human", "",
           f"Written by `tools/human_bar.py` ({meta['written']}). Source of the human results: Spire Codex public run export for v0.107.1 "
           f"(the four training fifths). Aggregates only; a cell over fewer than {MIN_CELL} runs is not shown.", "",
           f"- human starts the loop has played, with a result for the human: {meta['starts_played']:,} of {meta['human_starts']:,} "
           f"({meta['plays_used']:,} plays used: the latest {lists['n_last']} of each start with an outcome; {meta['plays_total']:,} plays in the log)",
           "- the player is the practice loop's x16 teacher with the newest checkpoint of the time, not the exam's champion; "
           "the exam's human stratum (`fight_exam.py report`) is the like-for-like reading of one player. The loop draws starts by "
           "regret (the ones it does worst at come up more often), so this table reads lower than an unselected sample would",
           "- a death costs the whole HP going in on both sides; value short is the human's result less the player's in the search's "
           "currency (end HP plus 10 a potion held, a death -100), averaged over starts: positive means the player fell short", "",
           "## By act and kind", ""]
    out += table_md(tab["by_act_kind"], "act and kind")
    out += ["", "## By enemy (the input to the solver and judge work)", ""]
    out += table_md(tab["by_enemy"], "act, kind, enemy")
    for name, key, what in (("Human survived, every one of the player's latest plays died", "human_survived_player_died",
                             f"starts with at least {lists['min_plays']} play(s) where the human survived the fight and all the player's latest plays died"),
                            (f"Player lost {lists['gap_hp']:g} HP or more above the human", "player_lost_more",
                             f"starts with at least {lists['min_plays']} play(s) where the mean HP the player lost exceeds the human's by {lists['gap_hp']:g} or more")):
        rows = by_enemy_counts(lists[key], figs)
        out += ["", f"## {name}", "", f"{len(lists[key]):,} {what}.", "", "| act, kind, enemy | on the list | played | share % |", "|---|---|---|---|"]
        hidden = 0
        for r in rows:
            if r["runs"] < MIN_CELL:
                hidden += 1
            else:
                out.append(f"| {r['cell']} | {r['on_list']:,} | {r['played']:,} | {100 * r['on_list'] / max(1, r['played']):.1f} |")
        if hidden:
            out += ["", f"{hidden} further cells are under {MIN_CELL} runs and not shown."]
    out += [""]
    return "\n".join(out)


def atomic_write(path: Path, text: str) -> None:
    path = Path(path)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(text, encoding="utf-8", newline="\n")
    os.replace(tmp, path)


def diagnose(root: Path, n_last: int = 4, gap_hp: float = 15.0, min_plays: int = 1, boot: int = 1000) -> tuple[dict, str]:
    """The diagnosis of the loop folder `root`: (the JSON document, the markdown)."""
    root = Path(root)
    lite = read_lite(root / "starts")
    outcomes = read_outcomes(root / "starts")
    rows = load_plays(root)
    plays = latest_plays(rows, n_last)
    figs = start_figures(plays, lite, outcomes)
    tab = tables(figs, boot)
    lists = short_lists(figs, gap_hp, min_plays, n_last)
    meta = {"rule": RULE, "written": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "human_starts": len(lite), "outcomes": len(outcomes),
            "plays_total": len(rows), "plays_used": sum(f["plays"] for f in figs), "starts_played": len(figs),
            "source": "Spire Codex public run export for v0.107.1 (the four training fifths)"}
    doc = {**meta, "table": tab, "lists": lists, "counts": {"human_survived_player_died": len(lists["human_survived_player_died"]),
                                                              "player_lost_more": len(lists["player_lost_more"])}}
    return doc, render_md(tab, lists, figs, meta)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--root", default=str(LOOP_ROOT))
    ap.add_argument("--out-dir", default=str(OUT_DIR))
    ap.add_argument("--n-last", type=int, default=int(pm.DEFAULTS["regret"]["n_last"]))
    ap.add_argument("--gap-hp", type=float, default=15.0)
    ap.add_argument("--min-plays", type=int, default=1)
    ap.add_argument("--boot", type=int, default=1000)
    ap.add_argument("--no-write", action="store_true")
    a = ap.parse_args(argv)
    doc, md = diagnose(Path(a.root), a.n_last, a.gap_hp, a.min_plays, a.boot)
    if not a.no_write:
        out = Path(a.out_dir)
        out.mkdir(parents=True, exist_ok=True)
        atomic_write(out / "diagnosis.json", json.dumps(doc, indent=1))
        atomic_write(out / "diagnosis.md", md)
    print(md)
    print(f"human_bar: {doc['starts_played']} human starts played; human survived and the player died: {doc['counts']['human_survived_player_died']}; "
          f"player lost {a.gap_hp:g}+ HP above the human: {doc['counts']['player_lost_more']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
