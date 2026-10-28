"""The human reference curve: Ironclad at Ascension 10 from the Codex corpus, AGGREGATES ONLY (D13).

    python -I tools/reference/codex_curve.py <runs.jsonl.gz> [--out docs/reference/codex-a10-ironclad]
        [--character IRONCLAD] [--ascension 10] [--min-cell 20] [--boot 2000]

The corpus is untrusted data from outside: one JSON object per human run, read once, line by line. Only tables of
counts, shares, means and quartiles leave this file, and a cell that rests on fewer than ``--min-cell`` runs is
suppressed (``null``), so nothing in the output identifies a run. No seed, run hash, player token, start time or
per-run row is ever kept past the pass, written, or logged. The writer refuses any output that holds a field name from
``FORBIDDEN_KEYS``.

Floors are the AI's floors: the map points of the run in order, counted across acts from 1 (the ancient is floor 1, the
Act 1 boss floor 17, the Act 2 boss 33, the Act 3 boss 48, the second Act 3 boss 49 and the win), the same numbers as
``rl/run_fields.ACT_BOSS_FLOORS`` and ``rl/gen_loop.boss_kill``.  The corpus keeps the room a run died in, so a death on
a boss has that boss's floor, as the AI's ``obs.floor`` does.

Standard library plus numpy (the bootstrap).  docs/reference/README.md maps each AI metric to the table it compares with.
"""

from __future__ import annotations

import argparse
import datetime
import gzip
import json
import re
import sys
from collections import Counter
from pathlib import Path

import numpy as np

FORMAT = 1
MIN_CELL = 20
BOOT = 2000
BOOT_SEED = 20261028
MAX_FLOOR = 57
# The floor of each act's boss, counted across acts (rl/run_fields.ACT_BOSS_FLOORS); the second Act 3 boss is one floor on.
ACT_BOSS_FLOORS = (17, 33, 48)
WIN_FLOOR = 49
VANILLA_ACTS = {"OVERGROWTH", "UNDERDOCKS", "HIVE", "GLORY"}
COMBAT_ROOMS = ("monster", "elite", "boss")
# Field names that must never reach an output (a run's identity or its time).
FORBIDDEN_KEYS = frozenset({"seed", "run_hash", "player_token", "start_time", "run_time", "player_id", "id", "players",
                            "platform_type", "badges"})

CHAR_RE = re.compile(rb'"character"\s*:\s*"CHARACTER\.([A-Z0-9_]+)"')
ASC_RE = re.compile(rb'"ascension"\s*:\s*(\d+)')
MODDED_ID = re.compile(rb'"(?:ACT|EVENT|ENCOUNTER|MONSTER|CARD|RELIC|POTION|ENCHANTMENT|MODIFIER|CHARACTER)\.[A-Z0-9_]*-')


# ------------------------------------------------------------------ one run

def bare(x) -> str:
    """``KIND.NAME`` -> ``NAME``."""
    return str(x).split(".", 1)[-1]


def points_of(run: dict) -> list:
    """The map points of the run in order, across acts: ``[(act, point), ...]``; floor f is element f - 1."""
    return [(a, mp) for a, act in enumerate(run.get("map_point_history") or []) for mp in act]


def stats_of(mp: dict) -> dict:
    ps = mp.get("player_stats") or [{}]
    return ps[0] if ps and isinstance(ps[0], dict) else {}


def room_of(mp: dict) -> dict:
    rooms = mp.get("rooms") or [{}]
    return rooms[0] if rooms and isinstance(rooms[0], dict) else {}


def summarize(run: dict) -> dict | None:
    """The per-run facts the tables are built from (kept in memory for the pass only); None for a run with no floors."""
    pts = points_of(run)
    n = len(pts)
    if n == 0:
        return None
    win = bool(run.get("win"))
    deck = (run.get("players") or [{}])[0].get("deck") or []
    # Every card the run ever held, with the floor it came in and the floor it left (None: still in the final deck).
    cards = [(int(c.get("floor_added_to_deck") or 1), None) for c in deck]
    picks = [0] * (n + 1)
    screens = [0] * (n + 1)
    rests = []
    for f, (act, mp) in enumerate(pts, 1):
        ps = stats_of(mp)
        for c in ps.get("cards_removed") or []:
            cards.append((int(c.get("floor_added_to_deck") or 1), f))
        for t in ps.get("cards_transformed") or []:
            o = t.get("original_card") or {}
            cards.append((int(o.get("floor_added_to_deck") or 1), f))
        ch = ps.get("card_choices") or []
        if ch and room_of(mp).get("room_type") in COMBAT_ROOMS:
            screens[f] = 1
            picks[f] = 1 if any(c.get("was_picked") for c in ch) else 0
        for r in ps.get("rest_site_choices") or []:
            rests.append((act, str(r).lower()))
    deck_size = [0] * (n + 1)
    delta = [0] * (n + 2)
    for added, left in cards:
        added = min(max(added, 1), n)
        delta[added] += 1
        if left is not None:
            delta[min(left, n)] -= 1  # a card removed on floor f no longer counts at the end of floor f
    run_total = 0
    for f in range(1, n + 1):
        run_total += delta[f]
        deck_size[f] = run_total
    # HP entering each act boss: the HP at the end of the floor before it.
    boss_hp = {}
    for f, (act, mp) in enumerate(pts, 1):
        if f > 1 and mp.get("map_point_type") == "boss" and pts[f - 2][0] == act and act < len(ACT_BOSS_FLOORS):
            prev = stats_of(pts[f - 2][1])
            if prev.get("max_hp"):
                boss_hp[act] = (int(prev.get("current_hp") or 0), int(prev["max_hp"]))
    end = None
    if not win:
        enc = str(run.get("killed_by_encounter") or "NONE.NONE")
        ev = str(run.get("killed_by_event") or "NONE.NONE")
        end = bare(enc) if enc != "NONE.NONE" else ("event:" + bare(ev) if ev != "NONE.NONE" else "unknown")
    return {"n": n, "win": win, "acts": len(run.get("map_point_history") or []), "deck": deck_size, "picks": picks,
            "screens": screens, "rests": rests, "boss_hp": boss_hp, "end": end,
            "end_act": pts[-1][0] if pts else None, "last_type": pts[-1][1].get("map_point_type"),
            "killer_in_last_room": (not win) and str(run.get("killed_by_encounter")) == str(room_of(pts[-1][1]).get("model_id"))}


def boss_killed(s: dict) -> list[bool]:
    """Act 1, Act 2, Act 3 first boss, Act 3 both bosses (the win): ``gen_loop.boss_kill``'s rule (the run won, or ended past
    the boss floor) applied at each boss floor.
    """
    return [s["win"] or s["n"] > b for b in ACT_BOSS_FLOORS] + [s["win"]]


def boss_killed_by_acts(s: dict) -> list[bool]:
    """The same by the corpus's own act lists: a run that reached act k+1 killed the boss of act k."""
    return [s["win"] or s["acts"] > 1, s["win"] or s["acts"] > 2, s["win"], s["win"]]


# ------------------------------------------------------------------ aggregation

def share(k: int, n: int, min_cell: int) -> dict:
    """A share cell, or suppressed (``None``) when it would rest on fewer than ``min_cell`` runs: the population n, the group k or
    the rest n - k is non-empty and below the floor.  Only the rounded share (0.1 point) is published, never n or k."""
    if n < min_cell or 0 < k < min_cell or 0 < n - k < min_cell:
        return {"share": None}
    return {"share": round(k / n, 3)}


def moments(xs: list, min_cell: int) -> dict:
    """Mean and quartiles over at least ``min_cell`` runs (the count itself is not published)."""
    if len(xs) < min_cell:
        return {"mean": None, "q1": None, "median": None, "q3": None}
    a = np.asarray(xs, dtype=np.float64)
    q1, med, q3 = (float(v) for v in np.percentile(a, [25, 50, 75]))
    return {"mean": round(float(a.mean()), 3), "q1": round(q1, 2), "median": round(med, 2), "q3": round(q3, 2)}


def bootstrap(floors: list, n_boot: int, seed: int = BOOT_SEED) -> dict:
    """Mean and median floor with the whole-run percentile bootstrap 95% interval (resampling runs, not floors)."""
    x = np.asarray(floors, dtype=np.float64)
    n = x.size
    if n == 0:
        return {"n": 0}
    rng = np.random.default_rng(seed)
    means = np.empty(n_boot)
    meds = np.empty(n_boot)
    step = max(1, 4_000_000 // n)
    for s in range(0, n_boot, step):
        e = min(n_boot, s + step)
        draw = x[rng.integers(0, n, size=(e - s, n))]
        means[s:e] = draw.mean(axis=1)
        meds[s:e] = np.median(draw, axis=1)
    lo, hi = np.percentile(means, [2.5, 97.5])
    mlo, mhi = np.percentile(meds, [2.5, 97.5])
    return {"n": int(n), "mean": round(float(x.mean()), 3), "mean_ci95": [round(float(lo), 3), round(float(hi), 3)],
            "median": float(np.median(x)), "median_ci95": [round(float(mlo), 2), round(float(mhi), 2)], "boot": n_boot}


def kaplan_meier(sums: list, aband: list) -> list:
    """The chance of getting past the Act 1 boss, the Act 2 boss and the last boss (the win) when an abandoned run is censored at
    the floor it left on (it is at risk up to there and never counts as a loss).  Product over floors of 1 - deaths / at risk."""
    runs = sums + aband
    top = max(s["n"] for s in runs)
    risk = np.bincount(np.asarray([s["n"] for s in runs], dtype=np.int64), minlength=top + 2)[::-1].cumsum()[::-1]
    dead = np.bincount(np.asarray([s["n"] for s in sums if not s["win"]], dtype=np.int64), minlength=top + 2)
    surv = np.ones(top + 2)
    for f in range(1, top + 1):
        surv[f] = surv[f - 1] * (1.0 - dead[f] / risk[f]) if risk[f] else surv[f - 1]
    return [float(surv[min(b, top)]) for b in (*ACT_BOSS_FLOORS[:2], WIN_FLOOR)]


def aggregate(sums: list, min_cell: int = MIN_CELL, n_boot: int = BOOT, abandoned: list | None = None) -> dict:
    """The tables over the summaries of the selected runs.  Every published cell is a rounded share or a moment over at least
    ``min_cell`` runs; no run count of a cell is published, and a floor of the survival curve is hidden when it differs from
    a neighbour by a count under ``min_cell`` (adjacent cells cannot be differenced to a handful of runs)."""
    n = len(sums)
    floors = [s["n"] for s in sums]
    out: dict = {"runs": n}
    if n == 0:
        return out
    # (a) survival: the share of runs reaching each floor (down to the last floor with enough runs), the share ending by each act boss
    cnt_ge = {f: sum(1 for x in floors if x >= f) for f in range(1, MAX_FLOOR + 2)}
    top = max((f for f in range(1, MAX_FLOOR + 1) if cnt_ge[f] >= min_cell), default=0)
    reach = {}
    for f in range(1, top + 1):
        cell = share(cnt_ge[f], n, min_cell)
        below = cnt_ge[f - 1] - cnt_ge[f] if f > 1 else 0
        above = cnt_ge[f] - cnt_ge[f + 1] if f < top else 0
        if 0 < below < min_cell or 0 < above < min_cell:
            cell = {"share": None}
        reach[str(f)] = cell
    out["survival"] = {"reach": reach}
    ended = {}
    for i, b in enumerate(ACT_BOSS_FLOORS, 1):
        ended[f"by_act{i}_boss"] = share(sum(1 for s in sums if not s["win"] and s["n"] <= b), n, min_cell)
        ended[f"at_act{i}_boss"] = share(sum(1 for s in sums if not s["win"] and s["n"] == b), n, min_cell)
    out["survival"]["ended"] = ended
    out["survival"]["died_in_act"] = {str(a + 1): share(sum(1 for s in sums if not s["win"] and s["end_act"] == a), n, min_cell)
                                      for a in range(3)}
    # (b) boss-kill rate per act (the AI's rule: win or ended past the boss floor) and the win rate
    kills = [boss_killed(s) for s in sums]
    by_act = [boss_killed_by_acts(s) for s in sums]
    reached = [sum(1 for s in sums if s["n"] >= b) for b in ACT_BOSS_FLOORS] + [sum(1 for s in sums if s["n"] >= WIN_FLOOR)]
    labels = ("act1", "act2", "act3_first", "act3_both")
    bk = {}
    for i, lab in enumerate(labels):
        k = sum(1 for r in kills if r[i])
        bk[lab] = {"killed_of_all": share(k, n, min_cell), "killed_of_reached": share(k, reached[i], min_cell),
                   "reached": share(reached[i], n, min_cell)}
    wins = sum(1 for s in sums if s["win"])
    out["boss_kill"] = {"rule": "the run won, or ended past the boss floor (gen_loop.boss_kill at each boss floor)", **bk,
                        "win": share(wins, n, min_cell),
                        "floor_rule_vs_act_lists_disagree": share(sum(1 for a, b in zip(kills, by_act) if a[:2] != b[:2]), n, min_cell)}
    # (c) floor reached
    out["floor"] = bootstrap(floors, n_boot)
    out["floor"]["by_outcome"] = {"died": moments([s["n"] for s in sums if not s["win"]], min_cell)}
    # (d) deck growth and card rewards per floor
    deck_by, scr_by, pick_by = {}, Counter(), Counter()
    for s in sums:
        for f in range(1, min(s["n"], top) + 1):
            deck_by.setdefault(f, []).append(s["deck"][f])
            scr_by[f] += s["screens"][f]
            pick_by[f] += s["picks"][f]
    out["deck"] = {"by_floor": {str(f): moments(deck_by[f], min_cell) for f in range(1, top + 1) if f in deck_by}}
    cards = {}
    for f in range(1, top + 1):
        sc, pk = scr_by[f], pick_by[f]
        if share(pk, sc, min_cell)["share"] is None:  # the same rule as every other cell: screens, takes and skips each 0 or >= min_cell
            cards[str(f)] = {"rewards_per_run": None, "taken_per_run": None, "skip_rate": None}
        else:
            cards[str(f)] = {"rewards_per_run": round(sc / cnt_ge[f], 3), "taken_per_run": round(pk / cnt_ge[f], 3),
                             "skip_rate": round(1 - pk / sc, 3)}
    tot_s, tot_p = sum(scr_by.values()), sum(pick_by.values())
    out["cards"] = {"by_floor": cards, "screen_rooms": list(COMBAT_ROOMS),
                    "taken_per_run": round(tot_p / n, 3), "screens_per_run": round(tot_s / n, 3),
                    "skip_rate": round(1 - tot_p / tot_s, 3) if tot_s >= min_cell else None}
    # (e) rest site choices by act
    rest = {}
    for a in [0, 1, 2, "all"]:
        c = Counter(r for s in sums for (act, r) in s["rests"] if a == "all" or act == a)
        tot = sum(c.values())
        rest["all" if a == "all" else str(a + 1)] = (
            {"choices": tot, "smith": share(c["smith"], tot, min_cell), "heal": share(c["heal"], tot, min_cell),
             "other": share(tot - c["smith"] - c["heal"], tot, min_cell)} if tot >= min_cell else {"choices": None})
    out["rests"] = rest
    # (f) HP entering each act boss
    hp = {}
    for a in range(3):
        v = [s["boss_hp"][a] for s in sums if a in s["boss_hp"]]
        hp[str(a + 1)] = {"hp": moments([x for x, _ in v], min_cell), "max_hp": moments([m for _, m in v], min_cell),
                          "hp_fraction": moments([x / m for x, m in v], min_cell)}
    out["boss_hp"] = hp
    # (g) end causes
    ends = Counter(s["end"] for s in sums if s["end"])
    dead = sum(ends.values())
    listed = [(k, v) for k, v in ends.most_common(40) if v >= min_cell and dead - v >= min_cell][:10]
    out["end_causes"] = {"dead_runs": dead if dead >= min_cell else None,
                         "top10": [{"cause": k, "share": round(v / dead, 3)} for k, v in listed],
                         "listed_share": round(sum(v for _, v in listed) / dead, 3) if dead >= min_cell else None}
    # checks that the floor numbering is the AI's
    out["checks"] = {"death_room_is_last_floor": share(sum(1 for s in sums if not s["win"] and s["killer_in_last_room"]),
                                                       sum(1 for s in sums if not s["win"]), min_cell)}
    # the abandoned runs: left out of the headline; the rates under three readings
    if abandoned is not None:
        na = len(abandoned)
        ok = na >= min_cell
        allk = [boss_killed(s) for s in abandoned]
        names = ("act1_kill", "act2_kill", "win")
        excluded = [bk["act1"]["killed_of_all"]["share"], bk["act2"]["killed_of_all"]["share"], out["boss_kill"]["win"]["share"]]
        lost = [round((sum(1 for r in kills if r[i]) + sum(1 for r in allk if r[i])) / (n + na), 3) for i in (0, 1, 3)] if ok else [None] * 3
        km = [round(v, 3) for v in kaplan_meier(sums, abandoned)] if ok else [None] * 3
        out["abandoned"] = {
            "reading": "left out of the headline: early quits (the player left the run), not a loss and not a statistical censoring",
            "share_of_selected": share(na, n + na, min_cell),
            "median_floor": float(np.median([s["n"] for s in abandoned])) if ok else None,
            "ended_by_floor_5": share(sum(1 for s in abandoned if s["n"] <= 5), na, min_cell),
            "won": share(sum(1 for s in abandoned if s["win"]), na, min_cell),
            "mean_floor_with_abandoned": round(float(np.mean(floors + [s["n"] for s in abandoned])), 3) if ok else None,
            "rates": {nm: {"excluded": excluded[j], "counted_as_lost": lost[j], "kaplan_meier": km[j]} for j, nm in enumerate(names)}}
    return out


# ------------------------------------------------------------------ the pass over the corpus

def lower_priority() -> None:
    """BelowNormal priority on Windows (a shared machine); nothing elsewhere."""
    if sys.platform != "win32":
        return
    try:
        import ctypes
        ctypes.windll.kernel32.SetPriorityClass(ctypes.windll.kernel32.GetCurrentProcess(), 0x00004000)
    except Exception:  # noqa: BLE001
        pass


def vanilla(run: dict) -> str | None:
    """Why a run is not a plain standard single-player run, or None."""
    if any(bare(a) not in VANILLA_ACTS for a in run.get("acts") or []):
        return "modded_act"
    if len(run.get("players") or []) != 1:
        return "not_single_player"
    if run.get("game_mode") != "standard":
        return "not_standard_mode"
    if run.get("modifiers"):
        return "modifiers"
    return None


def scan(lines, character: str = "IRONCLAD", ascension: int | None = 10) -> tuple[dict, list, list]:
    """One pass: the context counts, the summaries of the selected runs, the summaries of the abandoned ones."""
    lines_n = 0
    by_char: Counter = Counter()
    by_asc: Counter = Counter()
    by_char_asc: Counter = Counter()
    funnel: Counter = Counter()
    seen: set = set()
    sums: list = []
    aband: list = []
    builds: Counter = Counter()
    for line in lines:
        if not line.strip():
            continue
        lines_n += 1
        asc_m = ASC_RE.search(line)
        asc = int(asc_m.group(1)) if asc_m else -1
        chars = CHAR_RE.findall(line)
        ch = chars[0].decode() if len(chars) == 1 else ("MULTIPLAYER" if chars else "UNKNOWN")
        by_char[ch] += 1
        by_asc[asc] += 1
        by_char_asc[ch, asc] += 1
        if ch != character or (ascension is not None and asc != ascension):
            continue
        funnel["character_ascension"] += 1
        try:
            run = json.loads(line)
        except ValueError:
            funnel["unreadable"] += 1
            continue
        why = vanilla(run) or ("modded_content" if MODDED_ID.search(line) else None)
        if why:
            funnel[why] += 1
            continue
        key = run.get("run_hash") or f"{run.get('seed')}|{run.get('start_time')}"
        if key in seen:
            funnel["duplicate"] += 1
            continue
        seen.add(key)
        s = summarize(run)
        if s is None:
            funnel["no_floors"] += 1
            continue
        if run.get("was_abandoned"):
            funnel["abandoned"] += 1
            aband.append(s)
            continue
        builds[str(run.get("build_id"))] += 1
        funnel["analysed"] += 1
        sums.append(s)
    ctx = {"lines": lines_n, "by_character": dict(by_char), "by_ascension": {str(k): v for k, v in sorted(by_asc.items())},
           "by_character_ascension": {f"{c}|{a}": v for (c, a), v in sorted(by_char_asc.items())},
           "funnel": dict(funnel), "builds": dict(builds)}
    return ctx, sums, aband


VANILLA_CHARACTERS = ("IRONCLAD", "SILENT", "DEFECT", "NECROBINDER", "REGENT")
UNDER = "cells_under_floor"


def suppress_counts(ctx: dict, min_cell: int) -> dict:
    """The context counts with every cell below ``min_cell`` left out (the tables carry only how many cells that was, never their
    sizes) and, in the funnel and the builds, replaced by a ``<min_cell`` marker.  The character by ascension table keeps the
    game's own five characters only."""
    def drop(d):
        out = {k: v for k, v in d.items() if v >= min_cell}
        out[UNDER] = len(d) - (len(out))
        return out

    def mark(d):
        return {k: (v if v >= min_cell else f"<{min_cell}") for k, v in d.items()}

    keep = (*VANILLA_CHARACTERS, "MULTIPLAYER", "UNKNOWN")  # a third-party character is not named; all of them pool as OTHER
    chars = {k: v for k, v in ctx["by_character"].items() if k in keep}
    other = sum(v for k, v in ctx["by_character"].items() if k not in keep)
    if other:
        chars["OTHER"] = other
    by_char = dict(sorted(chars.items(), key=lambda kv: -kv[1]))
    cross = {k: v for k, v in ctx["by_character_ascension"].items() if k.split("|")[0] in VANILLA_CHARACTERS}
    return {"lines": ctx["lines"], "by_character": drop(by_char), "by_ascension": drop(ctx["by_ascension"]),
            "by_character_ascension": drop(cross), "funnel": mark(ctx["funnel"]), "builds": mark(ctx["builds"])}


def corpus_name(path: Path) -> str:
    m = re.search(r"runs-(v[0-9]+(?:\.[0-9]+)*)", path.name)
    return m.group(1) if m else path.name


def open_lines(path: Path):
    f = gzip.open(path, "rb") if path.suffix == ".gz" else Path(path).open("rb")
    with f:
        yield from f


def build(path: Path, character: str = "IRONCLAD", ascension: int | None = 10, min_cell: int = MIN_CELL, n_boot: int = BOOT,
          generated: str | None = None) -> dict:
    ctx, sums, aband = scan(open_lines(path), character, ascension)
    sha = None
    side = path.with_name(path.name + ".sha256")
    if side.exists():
        sha = side.read_text(encoding="utf-8").split()[0][:16]
    doc = {"format": FORMAT,
           "header": {"corpus": corpus_name(path), "corpus_sha256_prefix": sha,
                      "generated": generated or datetime.datetime.now(datetime.UTC).strftime("%Y-%m-%dT%H:%MZ"),
                      "selection": {"character": character, "ascension": ascension, "single_player": True, "standard_mode": True,
                                    "no_modifiers": True, "vanilla_content": True, "abandoned_runs": "excluded from the headline (early quits); rates under three readings in abandoned",
                                    "duplicates": "dropped"},
                      "min_cell": min_cell, "tool": "tools/reference/codex_curve.py"},
           "context": suppress_counts(ctx, min_cell),
           "reference": aggregate(sums, min_cell, n_boot, aband)}
    check_clean(doc)
    return doc


def check_clean(doc) -> None:
    """Refuse an output that holds a field of a run's identity."""
    stack = [doc]
    while stack:
        x = stack.pop()
        if isinstance(x, dict):
            bad = FORBIDDEN_KEYS & set(x)
            if bad:
                raise ValueError(f"output holds a forbidden field: {sorted(bad)}")
            stack.extend(x.values())
        elif isinstance(x, list):
            stack.extend(x)


# ------------------------------------------------------------------ Markdown

def pct(c, nd=1) -> str:
    return "-" if not c or c.get("share") is None else f"{100 * c['share']:.{nd}f}%"


def fmt(x, nd=1) -> str:
    return "-" if x is None else f"{x:.{nd}f}"


def rate(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}%"


def render_md(doc: dict) -> str:
    h, ref, ctx = doc["header"], doc["reference"], doc["context"]
    L = [f"# Human reference: {h['selection']['character'].title()} at Ascension {h['selection']['ascension']}", "",
         f"Corpus {h['corpus']} (sha256 prefix {h['corpus_sha256_prefix']}), generated {h['generated']} by `{h['tool']}`. "
         f"Aggregates only (D13): cells resting on fewer than {h['min_cell']} runs are shown as `-`; no run, seed or player is named. "
         "Shares are rounded to 0.1 point and carry no run counts; a survival floor that differs from a neighbour by under "
         f"{h['min_cell']} runs is hidden.", "", "## Runs", "", f"Lines read: {ctx['lines']}. Selection funnel:", ""]
    order = ("character_ascension", "modded_act", "not_single_player", "not_standard_mode", "modifiers", "modded_content", "duplicate",
             "no_floors", "unreadable", "abandoned", "analysed")
    fun = ctx["funnel"]
    L += ["| step | runs |", "|---|---|"] + [f"| {k} | {fun[k]} |" for k in order if k in fun] + [""]
    L += ["Runs by character (the game's five; first player of the run; `MULTIPLAYER` has several, `UNKNOWN` none the reader could find; "
          "`OTHER` pools every modded character):", "",
          "| character | runs |", "|---|---|"]
    L += [f"| {k} | {v} |" for k, v in ctx["by_character"].items() if k != UNDER]
    L += [f"| (listed rows under {h['min_cell']} runs) | {ctx['by_character'][UNDER]} |", ""]
    L += ["Runs by ascension:", "", "| ascension | runs |", "|---|---|"] + [f"| {k} | {v} |" for k, v in ctx["by_ascension"].items() if k != UNDER]
    L += [f"| (other ascensions, each under {h['min_cell']} runs) | {ctx['by_ascension'][UNDER]} ascensions |", ""]
    cross = {k: v for k, v in ctx["by_character_ascension"].items() if k != UNDER}
    asc_cols = sorted({int(k.split("|")[1]) for k in cross})
    chars = [c for c in VANILLA_CHARACTERS if any(k.startswith(c + "|") for k in cross)]
    L += ["Runs by character and ascension (the game's five characters; `-` under the floor):", "",
          "| ascension | " + " | ".join(chars) + " |", "|---|" + "---|" * len(chars)]
    L += [f"| {a} | " + " | ".join(str(cross.get(f"{c}|{a}", "-")) for c in chars) + " |" for a in asc_cols] + [""]
    if ref.get("runs", 0) == 0:
        return "\n".join(L) + "\nNo runs selected.\n"
    fl, bk, sv = ref["floor"], ref["boss_kill"], ref["survival"]
    L += ["## Headline", "", f"Analysed runs: {ref['runs']}. Abandoned runs are left out of this table (early quits; see the section on them "
          "for the rates with them counted).", "", "| metric | value |", "|---|---|",
          f"| mean floor | {fmt(fl['mean'], 2)} (95% CI {fl['mean_ci95'][0]:.2f} to {fl['mean_ci95'][1]:.2f}) |",
          f"| median floor | {fmt(fl['median'])} (95% CI {fl['median_ci95'][0]:.1f} to {fl['median_ci95'][1]:.1f}) |",
          f"| win rate | {pct(bk['win'])} |"]
    for lab in ("act1", "act2", "act3_first", "act3_both"):
        L.append(f"| boss kill {lab}, of all runs | {pct(bk[lab]['killed_of_all'])} |")
        L.append(f"| boss kill {lab}, of runs reaching it | {pct(bk[lab]['killed_of_reached'])} |")
    for a in ("1", "2", "3", "all"):
        r = ref["rests"][a]
        if r.get("choices"):
            L.append(f"| rest choices, act {a}: smith / heal / other | {pct(r['smith'])} / {pct(r['heal'])} / {pct(r['other'])} (of {r['choices']}) |")
    L += [f"| card rewards taken per run / skip rate | {ref['cards']['taken_per_run']} / {pct({'share': ref['cards']['skip_rate']})} |", ""]
    L += ["## Where runs end", "", "| act | died in act | died on its boss | ended by its boss |", "|---|---|---|---|"]
    for i in (1, 2, 3):
        L.append(f"| {i} | {pct(sv['died_in_act'][str(i)])} | {pct(sv['ended'][f'at_act{i}_boss'])} | {pct(sv['ended'][f'by_act{i}_boss'])} |")
    L += ["", "## HP entering each act boss", "", "| act | HP mean (q1 / median / q3) | max HP mean | HP share of max, mean |", "|---|---|---|---|"]
    for a in ("1", "2", "3"):
        b = ref["boss_hp"][a]
        L.append(f"| {a} | {fmt(b['hp']['mean'])} ({fmt(b['hp']['q1'], 0)} / {fmt(b['hp']['median'], 0)} / {fmt(b['hp']['q3'], 0)}) | "
                 f"{fmt(b['max_hp']['mean'])} | {fmt(b['hp_fraction']['mean'], 3)} |")
    L += ["", "## End causes (top 10 of runs that did not win)", "", "| cause | share of lost runs |", "|---|---|"]
    L += [f"| {t['cause']} | {100 * t['share']:.1f}% |" for t in ref["end_causes"]["top10"]] + [""]
    ab = ref.get("abandoned")
    if ab:
        L += ["## The abandoned runs", "",
              f"Abandoned runs are {pct(ab['share_of_selected'])} of the selected runs. They are early quits, not losses: median floor "
              f"{fmt(ab['median_floor'], 0)}, {pct(ab['ended_by_floor_5'])} end by floor 5, {pct(ab['won'])} won. The headline leaves them out; "
              "the rates under three readings (excluded, counted as lost at the floor they left on, and Kaplan-Meier with them censored at it):",
              "", "| rate | excluded (headline) | counted as lost | Kaplan-Meier censored |", "|---|---|---|---|"]
        for nm, lab in (("act1_kill", "Act 1 boss kill"), ("act2_kill", "Act 2 boss kill"), ("win", "win rate")):
            r = ab["rates"][nm]
            L.append(f"| {lab} | {rate(r['excluded'])} | {rate(r['counted_as_lost'])} | {rate(r['kaplan_meier'])} |")
        L += ["", f"Mean floor with them in: {fmt(ab['mean_floor_with_abandoned'], 2)}.", ""]
    L += ["## Per floor", "", "Floors are counted across acts (ancient = 1; Act 1 boss 17, Act 2 boss 33, Act 3 boss 48, win 49). "
          "Deck size is at the end of the floor; card rewards are the reward screens per run reaching the floor (monster, elite and boss rooms).", "",
          "| floor | reaching | deck mean (q1 / med / q3) | card rewards per run | skip rate |", "|---|---|---|---|---|"]
    for f, cell in sv["reach"].items():
        d = ref["deck"]["by_floor"].get(f, {})
        c = ref["cards"]["by_floor"].get(f, {})
        L.append(f"| {f} | {pct(cell)} | {fmt(d.get('mean'))} ({fmt(d.get('q1'), 0)} / {fmt(d.get('median'), 0)} / {fmt(d.get('q3'), 0)}) | "
                 f"{fmt(c.get('rewards_per_run'), 2)} | {pct({'share': c.get('skip_rate')})} |")
    return "\n".join(L) + "\n"


# ------------------------------------------------------------------ command line

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("corpus", type=Path)
    ap.add_argument("--out", type=Path, default=Path("docs/reference/codex-a10-ironclad"), help="output path without extension")
    ap.add_argument("--character", default="IRONCLAD")
    ap.add_argument("--ascension", type=int, default=10)
    ap.add_argument("--min-cell", type=int, default=MIN_CELL)
    ap.add_argument("--boot", type=int, default=BOOT)
    a = ap.parse_args(argv)
    lower_priority()
    doc = build(a.corpus, a.character, a.ascension, a.min_cell, a.boot)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.with_suffix(".json").write_text(json.dumps(doc, indent=1, sort_keys=False) + "\n", encoding="utf-8")
    a.out.with_suffix(".md").write_text(render_md(doc), encoding="utf-8")
    f = doc["context"]["funnel"]
    print(f"read {doc['context']['lines']} runs; analysed {f.get('analysed')}; wrote {a.out.with_suffix('.json')} and .md")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
