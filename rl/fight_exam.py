"""The fight exam (D55, item 3.1 of the 2026-11-04 train-and-test memo): a fixed, stratified, held-out set of fight starts, each
played on four paired draws, scored by more than win or loss. A parallel instrument beside the promotion board: it reads
`PROMOTION_RULE`'s games never and decides nothing; it prints what a candidate does to the champion's exam.

    python rl/fight_exam.py select   [--extra-starts PATH] [--per-cell 100] [--allow-practised] [--rebuild]   build the exam
    python rl/fight_exam.py add-human [--per-cell 50] [--total 400] [--rebuild]               add the human stratum (a second file)
    python rl/fight_exam.py add-act3  --starts-dir DIR [--rebuild] [--stand-alone]             add the Act 3 boss stratum (a third file)
    python rl/fight_exam.py baseline [--judge PATH]                                the judge's value of every opening position
    python rl/fight_exam.py play     --player v2 | --player-dir DIR [--half A|B|all] [--stratum all|ai|human] [--until STAMP]  one player on the exam
    python rl/fight_exam.py report   [--player v2]                                 the readings of one player's games
    python rl/fight_exam.py compare  CAND [--champion v2] [--no-play] [--until STAMP]   a candidate on half A against the champion

The set (`select`). Starts come from the gym's own start records (`gym-loop/starts`, the nights' replays) and, through
`--extra-starts PATH`, from a second file in the same record schema (id, cat, kind, act, floor, seed, encounter_id, hp, max_hp,
node_type, potions, doc; plus `stratum`, a label such as "human"). Strata: act (1, 2) by kind (hallway, elite, boss); a boss
is a cell of its own per boss (`cell_of`). Up to `--per-cell` (100) starts a cell, drawn by a seeded hash. A start is
*clean* when its run is in the loop's held-out split (never trained on) or the generation has never played it (`next_j` in
state.json is zero): only clean starts are taken, so the champion's first reading is not of positions it practised on; a cell
the clean pool cannot fill is reported short (`--allow-practised` tops it up with the least-played starts, marked
`practised`). Excluded: the promotion board's 200 starts and every run behind them (`gym_loop.board_run_seeds`, 338 runs). The
exam's own exclusion list is written to `exam/exclude_runs.json` (run seeds and start ids) for the practice sampler; the exam
keeps its own copy of every start's document (`exam/starts.jsonl`, the gym's format and byte index), so the gym's rebuilds of
its start set cannot move it. `select` refuses to replace an existing exam without `--rebuild`.

Draws. Four per start (`DRAWS`), game j of a start with seeds `fight_gym.game_seed(start id, j, NS + ...)` in the namespace
`NS` = "x" (the board's is "g", the practice's "L", the weights' "W"). Seeds never depend on the player, so a champion and a
candidate play the same shuffles and the same per-decision search seeds (common random numbers).

Halves (`half_of`). Half A or B by the first 32 bits of sha256("fight-exam-half:" + start id): below 2^31 is A. The gate reads
half A; half B is for the orchestrator's weekly read and no code path of the loop plays it. A start's half is stored in its record.

Scores. Per game: win (a capped game is lost), L0, HP lost (start HP minus end HP; the whole start HP in a loss), potions spent,
turns; and, from the position read before each decision (`Observer`; it never changes a move): enemy HP removed, energy spent on
cards, and block wasted (block held at the end of a turn above the attack the enemies' intents announce). Per start: the mean
over its draws. The baseline of a start is the judge's value of its opening positions (`baseline`: P(win) and L0 at each draw's
first decision, averaged over the draws), a property of the start and the shuffles, not of any player.

Regression adjustment (`adjusted_difference`). The paired difference of a start (candidate's mean minus champion's) is regressed
on the baseline's P(win) and P(win) x (1 - P(win)) over the starts compared, and the estimate is the mean difference moved to
the whole exam's mean baseline: it corrects half A's baseline imbalance against the exam it was cut from, and it takes out the
part of the differences that the start's difficulty explains. It is NOT AIVAT: the chance events after the first decision are
not corrected, and with common draws the start's difficulty already cancels in a paired difference, so the gain is the second
part of the sentence only. Both numbers are printed with a bootstrap interval over starts, resampled within cells.

Calibration flags (`calibration`). Starts the champion won on all four draws or on none are kept, weight zero in every statistic
here, and reported. A flag made from the champion's own draws and then used to compare it with a candidate on the same draws
selects the starts the champion happened to play at the middle of its range, which flatters the candidate when the start
difficulties are skewed; `compare` therefore also prints a cross-fitted figure (flags from draws 0 and 1, the difference on draws
2 and 3, then the other way round) which has no such selection.

The human stratum (`add-human`; D55 addendum 10). The human corpus's own result of a fight is the bar the player is held to: for each
rebuilt human fight start the sidecar `starts/human-outcomes.json` (rl/human_starts.py) holds whether the human survived the fight,
the HP they had left, the potions they used and the turns it took. `add-human` picks up to 50 *clean* starts a cell (act by kind, a
boss per boss; 400 in all: equal quotas, the remainder to cells in a fixed hash order), at most one per run, from the human starts
whose run the loop has never played (`state.json` `next_j` and `starts/plays.jsonl`), by the same seeded hash, and writes them as a
second file in the exam's folder (`starts-human.jsonl` with its index and lite list, `human-meta.json`). **Nothing of the AI exam is
rewritten**: its starts, cells, halves, meta and every reading of the champion on them stay as they were (the files' hashes are
checked before and after, and recorded). The runs of the human starts are added to `exclude_runs.json` (the practice drops them from the
next segment). `play` and `compare` play the human stratum with the rest (`--stratum ai|human` restricts); its games go in the same
games file (rows carry `stratum: "human"`), and every reading of the AI exam leaves them out. `report` and `compare` print the human
line after the exam's own: survival and HP lost of the human against the player's (the mean over its four draws), by act and kind (and
by boss), the paired difference of the two with a 95% bootstrap interval over starts (within cells), counts. Deaths count the whole HP going
in as lost, on both sides. No calibration flags are applied to it (the human is not a draw). The full exam is read by hand: no lab job
runs it, so the stratum adds its games (about 400 starts a half, four draws each) to the hand-run `play` and `compare`.

The human bar at a judgement (D55 addendum 11; report-only, no rule reads it). (1) The fight loop's gate (`gym_loop.human_bar`, daemon flag
`--human-bar on|off`) plays the stratum alone, 400 starts by four draws under `a3-v1`, for a candidate that reaches a stage-2 verdict, with
the gate's own processes and stop reasons (`play --player-dir DIR` is the same play by hand: a checkpoint folder is a player; its games go to
`games-<folder name>.jsonl`), and `human_bar_reading` / `human_bar_line` give the one line the gate logs and records: candidate, incumbent
and human survival (in 100) and HP lost on the same starts, and the candidate minus the incumbent paired by start with its 95% interval
(`cand 55.0 / inc 62.0 / human 79.8 in 100 on 400 starts; cand-inc -7.0 [-11.2, -2.8]; HP lost 41.2 / 38.6 / 33.2`); the incumbent's side is
left out when it has no games, and games on different engine tags are named. (2) The lab's keep test v2 plays nothing: `lab_human_bar` maps
the judged player's fight network (sha256 of its `combat_policy.pt`) to the gym player holding the same bytes (`gym_player_of`) and prints that
player's latest reading with the date of its games file, or says there are no human-stratum games for it; a network no gym player holds gets
the gym incumbent's reading, labelled as the incumbent's.

The Act 3 stratum (`add-act3`; the Act 3 week, lane 5). The two Act 3 bosses (floors 48 and 49) hold nine tenths of the gap to the average
human, and the Acts 1 and 2 exam had no cell for them. `rl/act3_starts.py` builds the positions (human openings rebuilt from the corpus's
training fifths, with the human's own result; our own arrivals from the plain runs' saved fight states); `add-act3` picks the exam's cells from
them: a boss per cell *and the boss's turn* (`act3-boss1-80`: the Test Subject as the first Act 3 boss; `act3-boss2-80`: as the second), with
quotas weighted to where our runs die (`act3_starts.HUMAN_QUOTA`, `OWN_QUOTA`), at most one start a run, the human side from runs the loop never
played. It is a third file in the exam's folder (`starts-act3.jsonl` with index and lite list, `act3-meta.json`); nothing of the AI exam or the
Acts 1 and 2 human stratum is rewritten (their files' hashes are checked before and after), the runs go on `exclude_runs.json` so the
practice's Act 3 stratum (`practice_mix`, off by default) never draws them, and `play`/`compare`/`report` leave the stratum alone (`--stratum
all` is the AI exam and the Acts 1 and 2 human stratum, as before). The stratum is played by `rl/act3_exam.py`: the champion's fight network on
the copy at the champion's own boss budget (the one the plain runs play), four draws a start, a report of kills by cell and by side (the human's
positions against ours) with the human bar beside.

The lock protocol. A run takes a `fight-exam` line in QUIET.lock (rewritten every ten minutes, removed at the end). It starts
only when no other tag but the loop's own (`gym-loop`, `gym-loop-train`) and `w3-blocks` stands, not while LAB/nightly/active
exists, in the night window, with LAB/HALT, LAB/PAUSE or (to start) main.lock; the loop sees the line, stops its running step and
waits; the run waits for the loop's line to go before it plays, and stops at once on any of those and at `--until`. Games
unfinished at a stop are lost and played again by the next run (a game with a row is never played twice). Processes are
BelowNormal and in a Job Object that ends with the run.

Lab code: nothing here changes the engine, the search or the solver. The human corpus's exam fifth (D38) is never an input.
"""

from __future__ import annotations

import argparse
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import random
import sys
import threading
import time
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fight_gym as fg  # noqa: E402
import gym_loop as gl  # noqa: E402
import practice_mix as pmix  # noqa: E402

# Nothing here loads the emulator library at import (the loop's rule): the pure parts run anywhere, the games in spawned processes.

EXAM_RULE = "fight-exam-v1"
TAG = "fight-exam"  # the run's QUIET.lock line
NS = "x"  # the draws' seed namespace (the board's is "g")
DRAWS = 4
PER_CELL = 100
SELECT_SEED = 20261104
EXTRA_PER_CELL = 100
HUMAN_RULE = "fight-exam-human-v1"
HUMAN_STEM = "starts-human"  # the human stratum's start file (and its index and lite list) beside the AI exam's
HUMAN_PER_CELL = 50
HUMAN_TOTAL = 400
HUMAN_SEED = 20261108
HUMAN_MIN_ROW = 20  # a per-boss line of the human reading is printed from this many starts
ALLOW_QUIET: list[str] = []  # QUIET.lock tags this run ignores besides its own (`--allow-quiet`, e.g. the lab's daytime score)
BOOT = 2000
BOOT_SEED = 20261105
HALF_PREFIX = "fight-exam-half:"
BEAT_S = 600.0  # the QUIET line is rewritten this often (the lab's runner reads a line older than 45 minutes as dead)
DEFAULT_CONFIG = "a3-v1"
EXAM_DIR = Path(os.environ.get("FIGHT_EXAM_DIR", str(gl.DEFAULT_ROOT / "exam")))
LOOP_ROOT = gl.DEFAULT_ROOT
# Names for display only (the env's encounter table, copied: the pure parts load no library).
ENCOUNTER_NAMES = {74: "ceremonial-beast", 77: "lagavulin-matriarch", 79: "soul-fysh", 82: "kin", 83: "vantom",
                   84: "waterfall-giant", 75: "kaiser-crab", 76: "knowledge-demon", 81: "insatiable",
                   73: "aeonglass", 78: "queen", 80: "test-subject"}
SENTINEL_HP = 100_000  # an enemy standing for a phase nothing can damage (a billion HP in the game's data) is not counted as HP to remove
MEASURES = ("won", "L0", "hp_lost", "potions_used", "turns", "dealt", "energy", "wasted")
CELL_ORDER = ("act1-hallway", "act1-elite", "act1-boss", "act2-hallway", "act2-elite", "act2-boss", "act3-boss")
ACT3_STEM = "starts-act3"  # the Act 3 stratum's start file (and its index and lite list) beside the AI exam's
ACT3_RULE = "fight-exam-act3-v1"
ACT3_SEED = 20261119


# ---------------------------------------------------------------------------------------------------------- the pure parts


def half_of(start_id: str) -> str:
    """The start's half: 'A' (the gate's) or 'B' (the weekly read), by a fixed hash of its id."""
    return "A" if int(hashlib.sha256((HALF_PREFIX + str(start_id)).encode()).hexdigest()[:8], 16) < 0x80000000 else "B"


def kind_cell(s: dict) -> str:
    """The act-and-kind stratum of a start: act1-hallway, act1-elite, act1-boss, act2-..."""
    return f"act{int(s['act'])}-{s['kind']}"


def cell_of(s: dict) -> str:
    """The cell: the stratum, and for a boss the boss too (act1-boss-74); a start that carries its boss's turn (`boss_no`, the Act 3 starts:
    the first of the two bosses or the second) puts it after the kind (act3-boss1-80)."""
    if s["kind"] != "boss":
        return kind_cell(s)
    no = s.get("boss_no")
    return f"{kind_cell(s)}{int(no)}-{s['encounter_id']}" if no else f"{kind_cell(s)}-{s['encounter_id']}"


def cell_label(cell: str) -> str:
    tail = cell.rsplit("-", 1)[-1]
    return f"{cell} ({ENCOUNTER_NAMES[int(tail)]})" if cell.count("-") == 2 and tail.isdigit() and int(tail) in ENCOUNTER_NAMES else cell


def order_key(seed: int, start_id: str) -> str:
    return hashlib.sha256(f"{seed}:{start_id}".encode()).hexdigest()


def is_clean(s: dict, practised: dict) -> bool:
    """A start the practice has never trained on: its run is in the trainer's held-out split, or the generation never played it."""
    return s.get("split") == "val" or int(practised.get(s["id"], 0)) == 0


def pick_exam(pool: list[dict], *, exclude_runs: set, practised: dict, per_cell: int = PER_CELL, seed: int = SELECT_SEED,
              allow_practised: bool = False) -> tuple[list[dict], dict]:
    """The exam's starts from the gym's start list: per cell, up to `per_cell` clean starts by the seeded hash (then, with
    `allow_practised`, the least-played of the others). Starts of an excluded run never. Returns (the chosen lite records with
    their cell, kind cell, half and `practised` count, a report by cell)."""
    cells: dict[str, list[dict]] = defaultdict(list)
    for s in pool:
        if s["seed"] in exclude_runs or not s.get("kind_same", True):
            continue
        cells[cell_of(s)].append(s)
    chosen: list[dict] = []
    report: dict[str, dict] = {}
    for c in sorted(cells):
        items = cells[c]
        clean = sorted((s for s in items if is_clean(s, practised)), key=lambda s: order_key(seed, s["id"]))
        rest = sorted((s for s in items if not is_clean(s, practised)), key=lambda s: (int(practised.get(s["id"], 0)), order_key(seed, s["id"])))
        take = clean[:per_cell]
        extra = rest[: max(0, per_cell - len(take))] if allow_practised else []
        take_ids = {s["id"] for s in take}
        for s in take + extra:
            chosen.append({**s, "cell": c, "kind_cell": kind_cell(s), "half": half_of(s["id"]), "practised": int(practised.get(s["id"], 0)),
                           "clean": s["id"] in take_ids, "stratum": s.get("stratum", "ai")})
        report[c] = {"pool": len(items), "clean": len(clean), "chosen": len(take) + len(extra), "practised_added": len(extra),
                     "short": len(take) + len(extra) < per_cell}
    return chosen, report


def pick_extra(records: list[dict], per_cell: int = EXTRA_PER_CELL, seed: int = SELECT_SEED) -> list[dict]:
    """The `--extra-starts` records the exam takes: up to `per_cell` a cell by the seeded hash. A record needs an id, a kind, an act,
    a seed, a document and (a boss) an encounter id; its stratum is its own label (default 'extra')."""
    need = ("id", "kind", "act", "seed", "doc")
    cells: dict[str, list[dict]] = defaultdict(list)
    seen: set = set()
    for r in records:
        miss = [k for k in need if r.get(k) is None] + (["encounter_id"] if r.get("kind") == "boss" and r.get("encounter_id") is None else [])
        if miss:
            raise ValueError(f"extra start {r.get('id')!r} lacks {miss}")
        if r["id"] in seen:
            raise ValueError(f"extra start id {r['id']!r} twice")
        seen.add(r["id"])
        cells[cell_of(r)].append(r)
    out = []
    for c in sorted(cells):
        for s in sorted(cells[c], key=lambda s: order_key(seed, s["id"]))[:per_cell]:
            out.append({**s, "cell": c, "kind_cell": kind_cell(s), "half": half_of(s["id"]), "practised": 0, "clean": True,
                        "stratum": s.get("stratum") or "extra", "cat": s.get("cat") or kind_cell(s)})
    return out


def exclusion(starts: list[dict]) -> dict:
    """The exam's exclusion list for the practice sampler: the runs behind its starts and the start ids."""
    return {"rule": EXAM_RULE, "run_seeds": sorted({str(s["seed"]) for s in starts}), "start_ids": sorted(s["id"] for s in starts),
            "starts": len(starts), "by_stratum": dict(Counter(s.get("stratum", "ai") for s in starts))}


def draw_seed(start_id: str, j: int, what: str = "load") -> int:
    """The seed of game j of a start in the exam's namespace; the player is not an input."""
    return fg.game_seed(start_id, j, NS + what)


def game_win(r: dict) -> bool:
    return bool(r.get("won")) and not r.get("capped")


def game_values(r: dict) -> dict:
    """One game's measures; missing ones (a capped row) are None."""
    def f(k):
        v = r.get(k)
        return None if v is None else float(v)
    return {"won": 1.0 if game_win(r) else 0.0, "L0": float(r.get("L0") or 0.0) if not r.get("capped") else 0.0,
            "hp_lost": f("hp_lost"), "potions_used": f("potions_used"), "turns": f("turns"), "dealt": f("dealt"),
            "energy": f("energy"), "wasted": f("wasted")}


def index_games(rows: list[dict]) -> dict:
    """{(start, j): row}, the last row of a game kept."""
    return {(r["start"], int(r["j"])): r for r in rows if "start" in r}


def mean(xs) -> float | None:
    xs = [x for x in xs if x is not None]
    return sum(xs) / len(xs) if xs else None


def start_means(games: dict, starts: dict, js: tuple | None = None) -> dict:
    """Per start, over its draws (all of `js` if given): {draws, and the mean of each measure over the games that have it}."""
    by: dict[str, list[dict]] = defaultdict(list)
    for (sid, j), r in games.items():
        if sid in starts and (js is None or j in js):
            by[sid].append(r)
    out = {}
    for sid, rs in by.items():
        vals = [game_values(r) for r in rs]
        out[sid] = {"draws": len(rs), **{m: mean(v[m] for v in vals) for m in MEASURES}, "dmg_per_energy": _ratio(vals, "dealt", "energy")}
    return out


def _ratio(vals: list[dict], num: str, den: str) -> float | None:
    n = sum(v[num] or 0.0 for v in vals if v[num] is not None)
    d = sum(v[den] or 0.0 for v in vals if v[den] is not None)
    return n / d if d > 0 else None


def calibration(games: dict, starts: dict, js: tuple = tuple(range(DRAWS))) -> dict:
    """The champion's calibration flags: starts it won on every draw of `js` ('always') or on none ('never'); the starts that lack
    one of those draws are 'unplayed' and not flagged. Weights: zero for always and never, one for the rest."""
    wins: dict[str, dict] = defaultdict(dict)
    for (sid, j), r in games.items():
        if sid in starts and j in js:
            wins[sid][j] = 1.0 if game_win(r) else 0.0
    full = {s: list(w.values()) for s, w in wins.items() if len(w) >= len(js)}
    always = sorted(s for s, w in full.items() if all(w))
    never = sorted(s for s, w in full.items() if not any(w))
    mixed = sorted(s for s, w in full.items() if any(w) and not all(w))
    return {"draws": len(js), "always": always, "never": never, "mixed": mixed, "unplayed": sorted(set(starts) - set(full))}


def weights_of(cal: dict, starts: dict) -> dict:
    """1.0 for every start the champion won sometimes but not always, 0.0 for always and never won (and for starts it has not played)."""
    mixed = set(cal["mixed"])
    return {sid: 1.0 if sid in mixed else 0.0 for sid in starts}


def percentile(sorted_xs: list[float], q: float) -> float:
    if not sorted_xs:
        return float("nan")
    i = q * (len(sorted_xs) - 1)
    lo = int(math.floor(i))
    hi = min(lo + 1, len(sorted_xs) - 1)
    return sorted_xs[lo] + (sorted_xs[hi] - sorted_xs[lo]) * (i - lo)


def bootstrap(groups: dict, stat, boot: int = BOOT, seed: int = BOOT_SEED) -> dict:
    """Percentile bootstrap over starts, resampled within their cells: `groups` {cell: [item, ...]}, `stat(groups)` a number.
    Returns {estimate, lo, hi, se, n}."""
    est = stat(groups)
    rnd = random.Random(seed)
    cells = {c: v for c, v in groups.items() if v}
    n = sum(len(v) for v in cells.values())
    draws = []
    for _ in range(boot if n >= 2 else 0):
        draws.append(stat({c: rnd.choices(v, k=len(v)) for c, v in cells.items()}))
    draws = sorted(x for x in draws if x is not None and not math.isnan(x))
    se = None
    if len(draws) > 1:
        m = sum(draws) / len(draws)
        se = math.sqrt(sum((x - m) ** 2 for x in draws) / (len(draws) - 1))
    return {"estimate": est, "lo": percentile(draws, 0.025) if draws else None, "hi": percentile(draws, 0.975) if draws else None,
            "se": se, "n": n}


def _flat(groups: dict, key: str) -> list[float]:
    return [x[key] for v in groups.values() for x in v if x.get(key) is not None]


def mean_stat(key: str):
    def stat(groups: dict) -> float | None:
        xs = _flat(groups, key)
        return sum(xs) / len(xs) if xs else None
    return stat


def paired_starts(champ: dict, cand: dict, starts: dict, weights: dict, js: tuple | None = None) -> dict:
    """The compared starts by cell: those with weight above zero that both played on the same draws (`js`, or the draws both
    have). Each item: the start and cand-minus-champion differences of each measure (`d_...`, with the two means beside) over
    those draws."""
    draws_of: dict[str, list[int]] = defaultdict(list)
    for (sid, j) in champ:
        if (sid, j) in cand and (js is None or j in js):
            draws_of[sid].append(j)
    out: dict[str, list[dict]] = defaultdict(list)
    for sid, s in starts.items():
        common = sorted(draws_of.get(sid, ()))
        if weights.get(sid, 0.0) <= 0.0 or not common:
            continue
        a = [game_values(champ[(sid, j)]) for j in common]
        b = [game_values(cand[(sid, j)]) for j in common]
        item = {"start": sid, "draws": len(common)}
        for m in MEASURES:
            ma, mb = mean(v[m] for v in a), mean(v[m] for v in b)
            item["d_" + m] = None if ma is None or mb is None else mb - ma
            item["champ_" + m], item["cand_" + m] = ma, mb
        item["d_dmg_per_energy"] = _dpe(b, a)
        out[s["cell"]].append(item)
    return out


def _dpe(b: list[dict], a: list[dict]) -> float | None:
    x, y = _ratio(b, "dealt", "energy"), _ratio(a, "dealt", "energy")
    return None if x is None or y is None else x - y


def mcnemar_counts(champ: dict, cand: dict, starts: dict, weights: dict, js: tuple | None = None) -> dict:
    """The games both players played on starts of weight above zero: candidate-only and champion-only wins, and the one-sided exact
    McNemar p (the loop's `mcnemar_one_sided`). Reported beside the promotion rule's, never in its place."""
    b = c = both = neither = 0
    for key in set(champ) & set(cand):
        sid, j = key
        if sid not in starts or weights.get(sid, 0.0) <= 0.0 or (js is not None and j not in js):
            continue
        x, y = game_win(cand[key]), game_win(champ[key])
        if x and not y:
            b += 1
        elif y and not x:
            c += 1
        elif x:
            both += 1
        else:
            neither += 1
    return {"cand_only": b, "champ_only": c, "both": both, "neither": neither, "pairs": b + c + both + neither,
            "p": gl.mcnemar_one_sided(b, c)}


# -- the baseline-adjusted difference


def features(x: float) -> list[float]:
    """The baseline's regressors: P(win) and P(win) (1 - P(win)) (the second is what a start that can still go either way has)."""
    return [x, x * (1.0 - x)]


def ols(rows: list[list[float]], y: list[float], ridge: float = 1e-9) -> list[float]:
    """Least squares of y on [1, row]: the coefficients (intercept first), by the normal equations with a tiny ridge."""
    p = len(rows[0]) + 1
    a = [[0.0] * p for _ in range(p)]
    b = [0.0] * p
    for r, t in zip(rows, y):
        v = [1.0, *r]
        for i in range(p):
            b[i] += v[i] * t
            for k in range(p):
                a[i][k] += v[i] * v[k]
    for i in range(p):
        a[i][i] += ridge * (1.0 + a[i][i])
    for i in range(p):  # Gauss-Jordan with partial pivoting
        piv = max(range(i, p), key=lambda r: abs(a[r][i]))
        a[i], a[piv] = a[piv], a[i]
        b[i], b[piv] = b[piv], b[i]
        d = a[i][i] or 1e-12
        for k in range(i, p):
            a[i][k] /= d
        b[i] /= d
        for r in range(p):
            if r != i and a[r][i]:
                f = a[r][i]
                for k in range(i, p):
                    a[r][k] -= f * a[i][k]
                b[r] -= f * b[i]
    return b


def adjusted_stat(key: str, ref: list[float]):
    """The statistic of the adjusted difference: fit d on the baseline's features over the sample, then the mean difference moved
    to the reference feature means `ref`. Items without a baseline (key 'x') or without the measure are dropped."""
    def stat(groups: dict) -> float | None:
        items = [it for v in groups.values() for it in v if it.get(key) is not None and it.get("x") is not None]
        if len(items) < 4:
            return None
        rows = [features(it["x"]) for it in items]
        beta = ols(rows, [it[key] for it in items])
        m = len(rows[0])
        fbar = [sum(r[i] for r in rows) / len(rows) for i in range(m)]
        dbar = sum(it[key] for it in items) / len(items)
        return dbar - sum(beta[1 + i] * (fbar[i] - ref[i]) for i in range(m))
    return stat


def attach_baseline(groups: dict, baseline: dict) -> None:
    """Put each start's baseline P(win) (mean over its draws) on its item as 'x'."""
    for v in groups.values():
        for it in v:
            b = baseline.get(it["start"])
            it["x"] = None if b is None else b["p_win"]


def reference_means(starts: dict, weights: dict, baseline: dict) -> list[float] | None:
    """The mean baseline features over the whole exam's starts that have weight (the population the adjustment moves to)."""
    xs = [baseline[s]["p_win"] for s in starts if weights.get(s, 0.0) > 0.0 and s in baseline]
    if not xs:
        return None
    f = [features(x) for x in xs]
    return [sum(r[i] for r in f) / len(f) for i in range(len(f[0]))]


def baseline_by_start(rows: list[dict]) -> dict:
    """{start: {p_win, l0, draws}}: the judge's opening values averaged over a start's draws."""
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["start"]].append(r)
    return {s: {"p_win": sum(x["p_win"] for x in v) / len(v), "l0": sum(x["l0"] for x in v) / len(v), "draws": len(v)} for s, v in by.items()}


def crossfit_difference(champ: dict, cand: dict, starts: dict, boot: int = BOOT) -> dict:
    """The paired win difference with flags that do not come from the draws it is measured on: starts flagged (all won or none won
    by the champion) on draws {0,1} are dropped from the difference on draws {2,3}, and the other way round; the two halves are
    pooled. The champion's four-draw flags select the starts it played in the middle of its range on those very draws."""
    pooled: dict[str, list[dict]] = defaultdict(list)
    for flag_js, eval_js in (((0, 1), (2, 3)), ((2, 3), (0, 1))):
        cal = calibration(champ, starts, js=flag_js)
        w = weights_of(cal, starts)
        part = paired_starts(champ, cand, starts, w, js=eval_js)
        for c, items in part.items():
            pooled[c].extend({**it, "start": f"{it['start']}#{eval_js[0]}"} for it in items)
    return bootstrap(pooled, mean_stat("d_won"), boot=boot)


# ---------------------------------------------------------------------------------------------------------- the exam's folder


def exam_paths(d: Path) -> dict:
    d = Path(d)
    return {"dir": d, "starts": d / "starts.jsonl", "meta": d / "meta.json", "exclude": d / "exclude_runs.json",
            "baseline": d / "baseline.jsonl", "log": d / "exam.log", "human_starts": d / f"{HUMAN_STEM}.jsonl",
            "human_lite": d / f"{HUMAN_STEM}.lite.json", "human_meta": d / "human-meta.json", "act3_starts": d / f"{ACT3_STEM}.jsonl",
            "act3_lite": d / f"{ACT3_STEM}.lite.json", "act3_meta": d / "act3-meta.json"}


def load_exam(d: Path = EXAM_DIR) -> dict:
    """{start id: lite record} of the exam (documents stay in starts.jsonl, read by index in the playing processes)."""
    p = exam_paths(d)
    lite = gl.read_json(p["starts"].with_name("starts.lite.json"), None)
    if lite is None:
        raise FileNotFoundError(f"no exam in {d}: run `fight_exam.py select`")
    return {s["id"]: s for s in lite}


def load_human_exam(d: Path = EXAM_DIR) -> dict:
    """{start id: lite record} of the human stratum (empty when `add-human` has not been run); the records carry the human's result."""
    lite = gl.read_json(exam_paths(d)["human_lite"], None)
    return {} if lite is None else {s["id"]: s for s in lite}


def load_act3_exam(d: Path = EXAM_DIR) -> dict:
    """{start id: lite record} of the Act 3 stratum (empty when `add-act3` has not been run); a human start carries the human's result."""
    lite = gl.read_json(exam_paths(d)["act3_lite"], None)
    return {} if lite is None else {s["id"]: s for s in lite}


def say(d: Path, msg: str) -> None:
    Path(d).mkdir(parents=True, exist_ok=True)
    with open(exam_paths(d)["log"], "a", encoding="utf-8") as fh:
        fh.write(f"{gl.stamp()} {gl.clean(msg)}\n")


def practised_counts(root: Path = LOOP_ROOT) -> dict:
    st = gl.read_json(Path(root) / "state.json", {}) or {}
    return {k: int(v) for k, v in (st.get("next_j") or {}).items()}


def build_exam(loop_root: Path = LOOP_ROOT, out: Path = EXAM_DIR, *, per_cell: int = PER_CELL, extra_path: Path | None = None,
               allow_practised: bool = False, rebuild: bool = False, seed: int = SELECT_SEED) -> dict:
    """Select the exam from the gym's start set and write it: starts.jsonl (+ index and lite list), meta.json and
    exclude_runs.json. Returns meta. An existing exam is kept unless `rebuild` (then its files get a dated backup name)."""
    p = exam_paths(out)
    if p["starts"].exists() and not rebuild:
        raise FileExistsError(f"{p['starts']} exists: the exam is a fixed ruler; --rebuild replaces it")
    loop_root = Path(loop_root)
    lite = gl.read_json(loop_root / "starts" / "starts.lite.json", [])
    if not lite:
        raise FileNotFoundError("no gym start set")
    board = gl.board_run_seeds()
    chosen, report = pick_exam(lite, exclude_runs=board, practised=practised_counts(loop_root), per_cell=per_cell, seed=seed,
                               allow_practised=allow_practised)
    src = loop_root / "starts" / "starts.jsonl"
    full = []
    for s in chosen:
        doc = gl.load_start(str(src), s["id"])
        full.append({**doc, **{k: s[k] for k in ("cell", "kind_cell", "half", "practised", "clean", "stratum")}})
    extras = []
    if extra_path is not None:
        extras = pick_extra(list(gl.iter_jsonl(Path(extra_path))))
        clash = {s["id"] for s in full} & {s["id"] for s in extras}
        if clash:
            raise ValueError(f"extra start ids clash with the gym's: {sorted(clash)[:3]}")
        full += extras
    if p["starts"].exists():
        stamp = gl.stamp().replace(":", "")
        for f in (p["starts"], p["starts"].with_name("starts.index.json"), p["starts"].with_name("starts.lite.json"), p["meta"], p["exclude"]):
            if f.exists():
                f.rename(f.with_name(f.name + f".{stamp}.bak"))
    p["dir"].mkdir(parents=True, exist_ok=True)
    gl.write_start_set(p["dir"], full)
    ex = exclusion(full)
    gl.write_json(p["exclude"], {**ex, "written": gl.stamp()})
    cells = defaultdict(lambda: {"pool": 0, "clean": 0, "chosen": 0, "short": False})
    cells.update(report)
    for s in extras:
        cells[s["cell"]]["extra"] = cells[s["cell"]].get("extra", 0) + 1
    by_cell = Counter(s["cell"] for s in full)
    by_kind = Counter(s["kind_cell"] for s in full)
    meta = {"rule": EXAM_RULE, "written": gl.stamp(), "seed": seed, "per_cell": per_cell, "draws": DRAWS, "namespace": NS,
            "half_rule": HALF_PREFIX + "<start id>: sha256, first 32 bits below 2^31 is A", "allow_practised": allow_practised,
            "starts": len(full), "runs": len(ex["run_seeds"]), "board_runs_excluded": len(board),
            "by_kind_cell": dict(sorted(by_kind.items())), "by_cell": {c: {**cells[c], "chosen_total": by_cell.get(c, 0)} for c in sorted(set(cells) | set(by_cell))},
            "short_cells": sorted(c for c, v in cells.items() if v.get("short")), "halves": dict(Counter(s["half"] for s in full)),
            "by_stratum": ex["by_stratum"], "practised_starts": sum(1 for s in full if not s["clean"]),
            "pool_starts": len(lite), "extra_file": extra_path is not None}
    gl.write_json(p["meta"], meta)
    say(out, f"select: {len(full)} starts, {meta['runs']} runs excluded from practice, short cells {meta['short_cells']}")
    return meta


# ---------------------------------------------------------------------------------------------------------- the human stratum


def played_runs(root: Path = LOOP_ROOT) -> set[str]:
    """The runs the loop has played a start of: those of `state.json`'s `next_j` entries above zero and of every start in
    `starts/plays.jsonl` (the plays log keeps what the buffer's retention removed). The evidence `select` uses for the gym's starts,
    by run: a start of a played run is no longer clean, because the practice learnt from its neighbour in the same run."""
    st = gl.read_json(Path(root) / "state.json", {}) or {}
    ids = {k for k, v in (st.get("next_j") or {}).items() if int(v) > 0}
    for r in gl.iter_jsonl(Path(root) / "starts" / "plays.jsonl"):
        if "start" in r:
            ids.add(r["start"])
    return {pmix.run_of(i) for i in ids}


def human_quota(avail: dict[str, int], per_cell: int = HUMAN_PER_CELL, total: int = HUMAN_TOTAL, seed: int = HUMAN_SEED) -> dict[str, int]:
    """Starts a cell: the same for every cell (the largest number up to `per_cell` that keeps the sum within `total`, a cell with fewer
    available taking all it has), then the remainder one start at a time to the cells that can take one, in a fixed hash order of
    the cell names."""
    cells = sorted(avail)
    q = max((q for q in range(per_cell + 1) if sum(min(q, avail[c]) for c in cells) <= total), default=0)
    quota = {c: min(q, avail[c]) for c in cells}
    left = total - sum(quota.values())
    order = sorted(cells, key=lambda c: hashlib.sha256(f"{seed}|{c}".encode()).hexdigest())
    while left > 0:
        grew = False
        for c in order:
            if left > 0 and quota[c] < min(per_cell, avail[c]):
                quota[c] += 1
                left -= 1
                grew = True
        if not grew:
            break
    return quota


def pick_human(pool: list[dict], outcomes: dict, *, played: set[str], per_cell: int = HUMAN_PER_CELL, total: int = HUMAN_TOTAL,
               seed: int = HUMAN_SEED) -> tuple[list[dict], dict, Counter]:
    """The human stratum's starts from the human start list: clean ones (their run never played by the loop) that have an outcome,
    by cell (act and kind, a boss per boss), at most one start a run, the `human_quota` of each cell by the seeded hash. Returns (the
    chosen lite records with their cell, half, `clean` and `stratum` and the human's result), a report by cell, and the count of
    starts left out by reason."""
    left_out: Counter = Counter()
    cells: dict[str, list[dict]] = defaultdict(list)
    for s in pool:
        if s["id"] not in outcomes:
            left_out["no outcome in the sidecar"] += 1
        elif {pmix.run_of(s["id"]), s.get("run"), s.get("seed")} & played:
            left_out["its run was played by the loop"] += 1
        else:
            cells[cell_of(s)].append(s)
    ordered = {c: sorted(v, key=lambda s: order_key(seed, s["id"])) for c, v in cells.items()}
    avail = {c: len({s["run"] for s in v}) for c, v in ordered.items()}
    quota = human_quota(avail, per_cell, total, seed)
    target = min(per_cell, total // max(1, len(ordered)))  # what every cell would have if the clean pool filled them all
    chosen: list[dict] = []
    report: dict[str, dict] = {}
    used: set[str] = set()
    for c in sorted(ordered):
        took = 0
        for s in ordered[c]:
            if took >= quota[c]:
                break
            if s["run"] in used:
                continue
            used.add(s["run"])
            took += 1
            chosen.append({**s, **outcomes[s["id"]], "cell": c, "kind_cell": kind_cell(s), "half": half_of(s["id"]), "practised": 0, "clean": True,
                           "stratum": "human"})
        report[c] = {"pool": len(ordered[c]), "runs": avail[c], "quota": quota[c], "chosen": took, "short": took < target}
    return chosen, report, left_out


def write_stem_set(d: Path, stem: str, starts: list[dict]) -> None:
    """`<stem>.jsonl` (one start a line with its document, sorted by id), its byte index and the list without documents: the shape of
    `gym_loop.write_start_set` under another name, so the AI exam's own files are not the ones written."""
    d = Path(d)
    starts = sorted(starts, key=lambda s: s["id"])
    index: dict[str, int] = {}
    out = d / f"{stem}.jsonl"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        for s in starts:
            index[s["id"]] = fh.tell()
            fh.write((json.dumps(s) + "\n").encode("utf-8"))
    os.replace(tmp, out)
    gl.write_json(d / f"{stem}.index.json", index)
    gl.write_json(d / f"{stem}.lite.json", [{k: v for k, v in s.items() if k != "doc"} for s in starts])


def ai_exam_files(p: dict) -> list[Path]:
    """The files that make the AI exam what it is (starts, index, lite list, meta, baseline): `add-human` leaves them as they are."""
    return [p["starts"], p["starts"].with_name("starts.index.json"), p["starts"].with_name("starts.lite.json"), p["meta"], p["baseline"]]


def build_human(loop_root: Path = LOOP_ROOT, out: Path = EXAM_DIR, *, starts_dir: Path | None = None, per_cell: int = HUMAN_PER_CELL,
                total: int = HUMAN_TOTAL, seed: int = HUMAN_SEED, rebuild: bool = False, exam_ids=None) -> dict:
    """Add the human stratum to the exam folder as a second file; returns its meta. The AI exam's files are checked unchanged, the
    human starts' runs go on `exclude_runs.json` (merged, the old file kept as `.pre-human.bak` once), and an existing stratum is kept
    unless `rebuild` (then its files get a dated backup name)."""
    import human_starts as hs  # noqa: PLC0415 -- pure Python; the corpus is not read here

    p = exam_paths(out)
    if not p["starts"].exists():
        raise FileNotFoundError(f"no exam in {out}: the human stratum is added to an exam")
    if p["human_starts"].exists() and not rebuild:
        raise FileExistsError(f"{p['human_starts']} exists: the stratum is a fixed ruler; --rebuild replaces it")
    loop_root = Path(loop_root)
    sd = Path(starts_dir) if starts_dir else loop_root / "starts"
    lite = gl.read_json(sd / f"{hs.STEM}.lite.json", [])
    side = gl.read_json(sd / hs.OUTCOME_FILE, {}) or {}
    outcomes = side.get("outcomes") or {}
    if not lite or not outcomes:
        raise FileNotFoundError(f"no human starts or no outcomes in {sd}: run `human_starts.py outcomes`")
    ids = hs.read_exam_ids() if exam_ids is None else frozenset(exam_ids)
    hs.assert_no_exam((s["run"] for s in lite), ids)  # the corpus's exam fifth is never an input
    played = played_runs(loop_root)
    chosen, report, left_out = pick_human(lite, outcomes, played=played, per_cell=per_cell, total=total, seed=seed)
    if not chosen:
        raise ValueError("no clean human start to add")
    hs.assert_no_exam((s["run"] for s in chosen), ids)
    before = {f.name: gl.sha256_file(f) for f in ai_exam_files(p) if f.exists()}
    src = str(sd / f"{hs.STEM}.jsonl")
    keys = ("cell", "kind_cell", "half", "practised", "clean", "stratum")
    full = [{**gl.load_start(src, s["id"]), **{k: s[k] for k in keys}, **outcomes[s["id"]]} for s in chosen]
    if p["human_starts"].exists():
        stamp = gl.stamp().replace(":", "")
        for f in (p["human_starts"], p["human_starts"].with_name(f"{HUMAN_STEM}.index.json"), p["human_lite"], p["human_meta"]):
            if f.exists():
                f.rename(f.with_name(f.name + f".{stamp}.bak"))
    write_stem_set(p["dir"], HUMAN_STEM, full)
    # the practice sampler drops the stratum's runs from its next segment
    ex = gl.read_json(p["exclude"], {}) or {}
    backup = p["exclude"].with_name(p["exclude"].name + ".pre-human.bak")
    if p["exclude"].exists() and not backup.exists():
        backup.write_bytes(p["exclude"].read_bytes())
    runs = sorted({s["run"] for s in full})
    ex["run_seeds"] = sorted(set(ex.get("run_seeds") or []) | set(runs))
    ex["start_ids"] = sorted(set(ex.get("start_ids") or []) | {s["id"] for s in full})
    ex["by_stratum"] = {**(ex.get("by_stratum") or {}), "human": len(full)}
    ex["starts"] = int(ex.get("starts") or 0) + len(full)
    ex["human_stratum"] = {"rule": HUMAN_RULE, "written": gl.stamp(), "starts": len(full), "runs": len(runs)}
    gl.write_json(p["exclude"], ex)
    after = {f.name: gl.sha256_file(f) for f in ai_exam_files(p) if f.exists()}
    if after != before:
        raise RuntimeError(f"the AI exam's files moved while the human stratum was added: {sorted(k for k in after if after[k] != before.get(k))}")
    by_cell = Counter(s["cell"] for s in full)
    meta = {"rule": HUMAN_RULE, "written": gl.stamp(), "seed": seed, "per_cell": per_cell, "total": total, "draws": DRAWS, "namespace": NS,
            "starts": len(full), "runs": len(runs), "halves": dict(Counter(s["half"] for s in full)),
            "by_kind_cell": dict(sorted(Counter(s["kind_cell"] for s in full).items())), "by_cell": {c: {**report[c], "chosen_total": by_cell.get(c, 0)} for c in sorted(report)},
            "short_cells": sorted(c for c, v in report.items() if v["short"]), "left_out": dict(left_out), "played_runs": len(played),
            "outcomes_rule": side.get("rule"), "outcomes_written": side.get("written"), "survived": sum(1 for s in full if s.get("human_survived")),
            "ai_files_sha256": before, "half_rule": HALF_PREFIX + "<start id>: the AI exam's rule"}
    gl.write_json(p["human_meta"], meta)
    say(out, f"add-human: {len(full)} human starts, {len(runs)} runs added to exclude_runs.json, short cells {meta['short_cells']}")
    return meta


def build_act3(out: Path = EXAM_DIR, *, starts_dir: Path, seed: int = ACT3_SEED, rebuild: bool = False, stand_alone: bool = False,
               human_quota: dict | None = None, own_quota: dict | None = None) -> dict:
    """Add the Act 3 stratum to the exam folder as a third file; returns its meta. `starts_dir` holds `human-starts-act3.*`,
    `human-outcomes-act3.json` and `own-starts-act3.*` (`rl/act3_starts.py`). The human side is the clean starts (their run never played by the
    loop, an outcome in the sidecar) by cell, at most one start a run across both sides; the own side is our arrivals by cell. The AI exam's
    files and the Acts 1 and 2 human stratum's are checked unchanged; the runs go on `exclude_runs.json` (merged, the old file kept as
    `.pre-act3.bak` once). `stand_alone`: the folder need not hold an AI exam (a staging folder). An existing stratum is kept unless `rebuild`."""
    import act3_starts as a3  # noqa: PLC0415 -- pure Python

    p = exam_paths(out)
    if not stand_alone and not p["starts"].exists():
        raise FileNotFoundError(f"no exam in {out}: the Act 3 stratum is added to an exam (--stand-alone for a staging folder)")
    if p["act3_starts"].exists() and not rebuild:
        raise FileExistsError(f"{p['act3_starts']} exists: the stratum is a fixed ruler; --rebuild replaces it")
    sd = Path(starts_dir)
    human_lite = gl.read_json(sd / f"{a3.HUMAN_STEM}.lite.json", [])
    outcomes = (gl.read_json(sd / a3.OUTCOME_FILE, {}) or {}).get("outcomes") or {}
    own_lite = gl.read_json(sd / f"{a3.OWN_STEM}.lite.json", [])
    if not human_lite or not outcomes:
        raise FileNotFoundError(f"no human Act 3 starts or no outcomes in {sd}: run `act3_starts.py human`")

    def human_skip(r: dict) -> str | None:
        return None if r["id"] in outcomes else "no outcome in the sidecar"

    taken: set = set()
    hq = a3.HUMAN_QUOTA if human_quota is None else human_quota
    oq = a3.OWN_QUOTA if own_quota is None else own_quota
    # clean first: a run the loop never played a start of; a cell the clean runs cannot fill is topped up with played runs, marked `practised`
    # (the practice never drew an Act 3 start, so what a played run carries over is its Acts 1 and 2 only)
    chosen_h, rep_h = a3.pick_quota(human_lite, hq, seed=seed, taken=taken, skip=human_skip, prefer=lambda r: not r.get("loop_played"))
    chosen_o, rep_o = a3.pick_quota(own_lite, oq, seed=seed, taken=taken) if own_lite else ([], {"_left_out": {"no own starts": 1}})
    if not chosen_h and not chosen_o:
        raise ValueError("no Act 3 start to add")
    guarded = ai_exam_files(p) + [p["human_starts"], p["human_starts"].with_name(f"{HUMAN_STEM}.index.json"), p["human_lite"], p["human_meta"]]
    before_files = [f for f in guarded if f.exists()]
    before = {f.name: gl.sha256_file(f) for f in before_files}
    full = []
    for stem, chosen in ((a3.HUMAN_STEM, chosen_h), (a3.OWN_STEM, chosen_o)):
        src = str(sd / f"{stem}.jsonl")
        for s in chosen:
            doc = gl.load_start(src, s["id"])
            extra = {"cell": s["cell"], "kind_cell": kind_cell(doc), "half": half_of(s["id"]), "practised": 0 if s["clean"] else 1, "clean": s["clean"],
                     "stratum": doc.get("stratum") or ("human" if stem == a3.HUMAN_STEM else "own")}
            if stem == a3.HUMAN_STEM:
                extra.update(outcomes[s["id"]])
            full.append({**doc, **extra})
    if p["act3_starts"].exists():
        stamp = gl.stamp().replace(":", "")
        for f in (p["act3_starts"], p["act3_starts"].with_name(f"{ACT3_STEM}.index.json"), p["act3_lite"], p["act3_meta"]):
            if f.exists():
                f.rename(f.with_name(f.name + f".{stamp}.bak"))
    p["dir"].mkdir(parents=True, exist_ok=True)
    write_stem_set(p["dir"], ACT3_STEM, full)
    ex = gl.read_json(p["exclude"], {}) or {}
    backup = p["exclude"].with_name(p["exclude"].name + ".pre-act3.bak")
    if p["exclude"].exists() and not backup.exists():
        backup.write_bytes(p["exclude"].read_bytes())
    runs = sorted({s["run"] for s in full})
    ex["run_seeds"] = sorted(set(ex.get("run_seeds") or []) | set(runs))
    ex["start_ids"] = sorted(set(ex.get("start_ids") or []) | {s["id"] for s in full})
    ex["by_stratum"] = {**(ex.get("by_stratum") or {}), "act3-human": len(chosen_h), "act3-own": len(chosen_o)}
    ex["starts"] = int(ex.get("starts") or 0) + len(full)
    ex["act3_stratum"] = {"rule": ACT3_RULE, "written": gl.stamp(), "starts": len(full), "runs": len(runs)}
    gl.write_json(p["exclude"], ex)
    after = {f.name: gl.sha256_file(f) for f in before_files if f.exists()}
    if after != before:
        raise RuntimeError(f"the exam's files moved while the Act 3 stratum was added: {sorted(k for k in after if after[k] != before.get(k))}")
    by_cell = Counter(s["cell"] for s in full)
    meta = {"rule": ACT3_RULE, "written": gl.stamp(), "seed": seed, "draws": DRAWS, "namespace": NS, "starts": len(full), "runs": len(runs),
            "halves": dict(Counter(s["half"] for s in full)), "by_stratum": dict(Counter(s["stratum"] for s in full)),
            "by_cell": {c: {"chosen_total": by_cell.get(c, 0), "human": dict(rep_h.get(c, {})), "own": dict(rep_o.get(c, {}))}
                        for c in sorted(set(hq) | set(oq))},
            "left_out": {"human": rep_h.get("_left_out"), "own": rep_o.get("_left_out")},
            "practised_starts": sum(1 for s in full if not s["clean"]),
            "human_survived": sum(1 for s in full if s.get("stratum") == "human" and s.get("human_survived")),
            "quota": {"human": hq, "own": oq}, "files_sha256": before}
    gl.write_json(p["act3_meta"], meta)
    say(out, f"add-act3: {len(chosen_h)} human and {len(chosen_o)} own starts, {len(runs)} runs added to exclude_runs.json")
    return meta


# ---------------------------------------------------------------------------------------------------------- the games (child side)


class Observer:
    """`play_steps`'s grader hook: reads the position before each decision (HP, block, energy, the enemies' HP and announced
    attack) and from the step between two decisions takes the enemy HP removed and the energy spent on a card; at each end of turn
    the block held above the announced attack is wasted block. It returns an empty grade and never changes the move. A position the
    engine will not write out (a card's choice screen in the middle of a step) is skipped, counted in `misses`, and the next reading
    carries the change over the gap; an enemy of a billion HP (an unkillable phase) is not counted. Nothing it does can end a game."""

    def __init__(self) -> None:
        self.prev: tuple | None = None
        self.dealt = 0.0
        self.energy = 0.0
        self.wasted = 0.0
        self.block_end = 0.0
        self.ends = 0
        self.looks = 0
        self.misses = 0

    @staticmethod
    def look(world) -> dict:
        from sts2_gym import native  # noqa: PLC0415

        return digest(native.get_state_json(world.handle))

    def account(self, cur: dict) -> None:
        if self.prev is None:
            return
        prev, kind = self.prev
        self.dealt += max(0.0, prev["ehp"] - cur["ehp"])
        if kind == "card" and cur["turn"] == prev["turn"]:
            self.energy += max(0.0, prev["energy"] - cur["energy"])

    def __call__(self, world, k: int, best) -> dict:
        try:
            cur = self.look(world)
        except Exception:  # noqa: BLE001 -- a position the engine will not write out (a card's choice screen mid-step): no reading
            self.misses += 1
            return {}
        self.looks += 1
        self.account(cur)
        if best.kind == "end_turn":
            self.ends += 1
            self.block_end += cur["block"]
            self.wasted += max(0.0, cur["block"] - cur["incoming"])
        self.prev = (cur, best.kind)
        return {}

    def finish(self, world, won: bool) -> None:
        """The step after the last decision: a won fight removed what the enemies had left."""
        if self.prev is None:
            return
        if won:
            self.dealt += max(0.0, self.prev[0]["ehp"])
            return
        with contextlib.suppress(Exception):
            self.account(self.look(world))

    def summary(self) -> dict:
        return {"dealt": round(self.dealt, 1), "energy": round(self.energy, 1), "wasted": round(self.wasted, 1),
                "block_end": round(self.block_end, 1), "end_turns": self.ends, "looks": self.looks, "look_misses": self.misses}


def digest(doc: dict) -> dict:
    """The few numbers of a state document the observer reads: the player's HP, block and energy, the turn, the enemies' HP, and the
    damage the enemies' intents announce (an attack intent's magnitude times its hits)."""
    p = doc.get("player") or {}
    enemies = doc.get("enemies") or []
    inc = 0
    for e in enemies:
        it = e.get("intent") or {}
        if int(it.get("type", -1)) == 0 or it.get("carries_damage"):
            inc += int(it.get("magnitude") or 0) * max(1, int(it.get("hits") or 1))
    return {"hp": int(p.get("hp", 0)), "block": int(p.get("block", 0)), "energy": int(p.get("energy", 0)), "turn": int(doc.get("turn", 0)),
            "ehp": sum(int(e.get("hp", 0)) for e in enemies if 0 < int(e.get("hp", 0)) < SENTINEL_HP), "incoming": inc}


_ENGINE: list = [None]


def engine_tag_cached() -> str:
    """The sha256 prefix of the engine library this process runs (the gate's `engine_tag`), computed once."""
    if _ENGINE[0] is None:
        try:
            from sts2_gym import native  # noqa: PLC0415

            _ENGINE[0] = gl.sha256_file(Path(native._lib._name))[:12]  # noqa: SLF001
        except (ImportError, AttributeError, OSError):
            _ENGINE[0] = "unknown"
    return _ENGINE[0]


def exam_game_steps(task: tuple, clock, game, ctx):
    """One exam game as the lockstep's game steps: `fight_gym.play_steps` in the exam's namespace under the task's config, the
    observer as its grader. task: (arm for the deadline, start id, j, cat, info) with info {starts, player, config, ns, label}.
    The move played is the gym's: the observer only reads."""
    arm, sid, j, cat, info = task
    start = gl.load_start(info["starts"], sid)
    world = fg._fresh_world(start["doc"], sid, j, info["ns"])  # noqa: SLF001
    obs = Observer()
    try:
        row = yield from fg.play_steps(world, start, j, gl.player_config(info["config"]), ctx.extra, info["ns"],
                                       deadline=fg.task_deadline(task), clock=clock, progress=game.note, grader=obs)
        obs.finish(world, bool(row["won"]))
    finally:
        world.close()
    row.pop("grades", None)
    hp0 = int(start.get("hp") or 0)
    out = {"_task": [arm, sid, j], "arm": info.get("label", arm), "config": info["config"], "start": sid, "j": j, "cat": cat,
           "cell": start.get("cell"), "half": start.get("half"), "stratum": start.get("stratum"), "player": info["player"],
           "seed": start.get("seed"), "start_hp": hp0, "start_potions": start.get("potions"), "start_max_hp": start.get("max_hp"),
           "exam": EXAM_RULE, "engine": engine_tag_cached(), **row, **obs.summary()}
    out["hp_lost"] = hp0 - (int(row["end_hp"]) if row["won"] else 0)
    return out


# ---------------------------------------------------------------------------------------------------------- the lock


class Blocked(Exception):
    pass


def _lines(p: Path) -> list[str]:
    return gl._lines(p)  # noqa: SLF001


def blocked_reason(*, starting: bool, until: str | None = None, stop_file: Path | None = None) -> str | None:
    """Why the exam run must not compute now (None: it may). The night (active file or window), HALT, PAUSE, a live QUIET.lock line
    of anyone but the loop's own tags, w3-blocks and our own; main.lock stops a start only; `until` is a UTC stamp."""
    if stop_file is not None and stop_file.exists():
        return "stop"
    if until and gl.stamp() >= until:
        return "until"
    if gl.HALT.exists():
        return "halt"
    if fg.ACTIVE.exists():
        return "nightly-active"
    if gl.in_night_window():
        return "night-window"
    if gl.PAUSE.exists():
        return "pause"
    if starting and fg.MAIN_LOCK.exists():
        return "main.lock"
    tag = gl.other_quiet(allow=(TAG, *ALLOW_QUIET))
    return f"quiet:{tag}" if tag else None


def hold_line(on: bool) -> None:
    """Hold (or drop) the exam run's QUIET.lock line with its Windows PID; every other line is kept as it is."""
    keep = [x for x in _lines(fg.QUIET) if x.split()[0] != TAG]
    if on:
        keep.append(f"{TAG} {gl.stamp()} pid={os.getpid()}")
    if keep:
        fg.QUIET.write_text("".join(x + "\n" for x in keep), encoding="utf-8", newline="\n")
    else:
        fg.QUIET.unlink(missing_ok=True)


class ExamLock:
    """The run's claim on the machine: refused when anything blocks a start; else the line is written, kept alive by a heartbeat, and
    the run waits (at most `wait_s`) for the loop's own line to go, which it does within a step's stop poll; dropped on exit."""

    def __init__(self, until: str | None = None, stop_file: Path | None = None, wait_s: float = 600.0) -> None:
        self.until, self.stop_file, self.wait_s = until, stop_file, wait_s
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def __enter__(self) -> ExamLock:
        why = blocked_reason(starting=True, until=self.until, stop_file=self.stop_file)
        if why:
            raise Blocked(why)
        hold_line(True)
        end = time.time() + self.wait_s
        while any(x.split()[0] == gl.TAG for x in _lines(fg.QUIET)) and time.time() < end:
            time.sleep(5)
            why = blocked_reason(starting=False, until=self.until, stop_file=self.stop_file)
            if why:
                hold_line(False)
                raise Blocked(why)
        if any(x.split()[0] == gl.TAG for x in _lines(fg.QUIET)):
            hold_line(False)
            raise Blocked("the loop's line did not go")
        time.sleep(10)  # its servers are down by now (its line goes after them)
        self._thread = threading.Thread(target=self._beat, daemon=True)
        self._thread.start()
        return self

    def _beat(self) -> None:
        while not self._stop.wait(BEAT_S):
            with contextlib.suppress(OSError):
                hold_line(True)

    def __exit__(self, *exc) -> None:
        self._stop.set()
        hold_line(False)

    def stop_reason(self) -> str | None:
        return blocked_reason(starting=False, until=self.until, stop_file=self.stop_file)


class _Shim:
    """What `gym_loop.Server` reads of a Loop: the folder to keep the server's stats in, and a log."""

    def __init__(self, root: Path) -> None:
        self.root = Path(root)
        self.games = self.root / "games"
        self.games.mkdir(parents=True, exist_ok=True)

    def log(self, msg: str) -> None:
        say(self.root, msg)


# ---------------------------------------------------------------------------------------------------------- playing


def resolve_player(name: str, loop_root: Path = LOOP_ROOT) -> Path:
    """A player folder: a path, or a name under the loop's players/ then checkpoints/."""
    p = Path(name)
    if p.is_dir():
        return p
    for sub in ("players", "checkpoints"):
        q = Path(loop_root) / sub / name
        if q.is_dir():
            return q
    raise FileNotFoundError(f"no player {name!r}")


def exam_tasks(starts: dict, label: str, player: str, config: str, exam_dir: Path, done: set, draws: int = DRAWS,
               half: str = "all", limit: int = 0, human: dict | None = None) -> list:
    """The games still to play, long fights first (bosses, then elites, then hallways: the run's tail is the short ones). The human
    stratum's starts (`human`) are played from their own file. `limit` (a smoke test) keeps that many starts, spread over the cells by
    the seeded hash."""
    p = exam_paths(exam_dir)
    base = {"player": player, "config": config, "ns": NS, "label": label}
    infos = {"ai": {"starts": str(p["starts"]), **base}, "human": {"starts": str(p["human_starts"]), **base}}
    rank = {"boss": 0, "elite": 1, "hallway": 2}
    sel = [(s, infos["ai"]) for s in starts.values() if half == "all" or s["half"] == half]
    sel += [(s, infos["human"]) for s in (human or {}).values() if half == "all" or s["half"] == half]
    if limit:
        sel = sorted(sel, key=lambda x: order_key(SELECT_SEED, x[0]["id"]))[:limit]
    tasks = [("a3-v1", s["id"], j, s["cat"], info) for s, info in sel for j in range(draws) if (s["id"], j) not in done]
    kind = {s["id"]: s["kind"] for s, _ in sel}
    tasks.sort(key=lambda t: (rank[kind[t[1]]], t[2], t[1]))
    return tasks


def play_exam_tasks(exam_dir: Path, lock: ExamLock, tasks: list, target: Path, player_dir: Path, what: str, *, procs: int = 6, k: int = 32,
                    prior: str = "server", judge: str | None = None) -> str | None:
    """Play `tasks` in the loop's lockstep processes (the prior on the card when GPU_OK allows), rows appended to `target`. Returns
    why it stopped, or None when every task has a row. The caller holds the lock."""
    import gym_lockstep  # noqa: PLC0415

    shim = _Shim(exam_dir)
    init_args = (gl.ENGINE, str(player_dir.parent), player_dir.name, str(shim.root))
    server = gl.Server(shim, player_dir, what) if prior == "server" else None
    opts = {"prior": "server" if server is not None and server.opts else "cpu", "server": server.opts if server else None,
            "game_steps": "fight_exam:exam_game_steps", "worlds": "reload", "grade_wall_s": 0, "paths": [str(ROOT / "rl")]}
    judge_server = None
    if judge:
        j = {"path": str(judge), "sha256": gl.sha256_file(Path(judge))}
        if server is not None and server.opts:
            from prior_launch import start_prior_server  # noqa: PLC0415

            judge_server = start_prior_server(gl.GPU_PYTHON, j["path"], lib_path=ROOT / "out", kill_file=gl.GPU_OFF,
                                              stats_path=shim.games / f"judge-server-{what}.json", priority="below")
            if judge_server is not None:
                j.update(endpoint=judge_server.endpoint, authkey=judge_server.authkey)
        opts["judge"] = j
    t0 = time.perf_counter()
    try:
        stopped, stats = gym_lockstep.run_lockstep(tasks, target, procs, min(k, gym_lockstep.K_MAX), init_args, opts, stop=lock.stop_reason,
                                                   log=lambda m: say(exam_dir, f"{what}: {m}"))
    finally:
        if judge_server is not None:
            judge_server.stop()
        sstats = server.stop() if server is not None else {}
    say(exam_dir, f"{what}: {len(tasks)} games asked, prior {opts['prior']}, {time.perf_counter() - t0:.0f} s, stopped {stopped}, server "
        f"{ {x: sstats.get(x) for x in ('replays', 'rows', 'busy_s') if x in sstats} }")
    return stopped


def prepare_process() -> None:
    """BelowNormal for this process and a Job Object that ends everything it starts with it; the one-exam-at-a-time mutex."""
    from tools.lab import winsys  # noqa: PLC0415

    winsys.set_below_normal()
    if not winsys.acquire_mutex("spireline-fight-exam"):
        raise Blocked("another fight exam run is on")
    try:
        job = winsys.JobObject(kill_on_close=True)
        job.assign_self()
        prepare_process.job = job  # type: ignore[attr-defined]
    except OSError as error:
        say(EXAM_DIR, f"job object: {error}")


def run_player(a: argparse.Namespace, label: str, player_dir: Path, half: str, games: Path) -> int:
    """Play a player's missing exam games under the lock. Returns 0 (all played), 3 (blocked or stopped before the end)."""
    d = Path(a.exam_dir)
    stratum = getattr(a, "stratum", "all")
    all_starts = load_exam(d)
    starts = all_starts if stratum in ("all", "ai") else {}
    human = load_human_exam(d) if stratum in ("all", "human") else {}
    if stratum == "human" and not human:
        print(f"{label}: no human stratum in {d}: run `fight_exam.py add-human`")
        return 2
    done = {(r["start"], int(r["j"])) for r in fg.read_jsonl(games)}
    draws, limit = getattr(a, "draws", DRAWS), getattr(a, "limit", 0)
    tasks = exam_tasks(starts, label, label, a.config, d, done, draws=draws, half=half, limit=limit, human=human)
    if not tasks:
        print(f"{label}: all {len(done)} games are played")
        return 0
    prepare_process()
    until = a.until or None
    try:
        with ExamLock(until=until, stop_file=d / "STOP") as lock:
            if not (exam_paths(d)["baseline"]).is_file() and not a.no_baseline:
                make_baseline(d, a.judge_path, all_starts)
            say(d, f"{label}: playing {len(tasks)} games of {len(starts)} starts"
                   f"{f' and {len(human)} human starts' if human else ''} (half {half}) under {a.config}")
            stopped = play_exam_tasks(d, lock, tasks, games, player_dir, label, procs=a.procs, k=a.k, prior=a.prior, judge=a.judge or None)
    except Blocked as why:
        print(f"{label}: not started: {why}")
        say(d, f"{label}: not started: {why}")
        return 3
    left = len(exam_tasks(starts, label, label, a.config, d, {(r["start"], int(r["j"])) for r in fg.read_jsonl(games)}, draws=draws,
                          half=half, limit=limit, human=human))
    print(f"{label}: stopped ({stopped}); {left} games left" if stopped else f"{label}: played; {left} left")
    return 3 if stopped or left else 0


# ---------------------------------------------------------------------------------------------------------- the baseline


def make_baseline(exam_dir: Path, judge_path: str, starts: dict | None = None, draws: int = DRAWS) -> int:
    """The judge's value of every opening position: for each start and draw, P(win) and L0 at the first decision of the draw's world
    (CPU, one process). Appends to baseline.jsonl, skipping what is there. Runs under the caller's lock."""
    import gen0  # noqa: PLC0415

    gen0.select_engine(gl.ENGINE)
    import judge_client  # noqa: PLC0415

    starts = starts or load_exam(exam_dir)
    path = exam_paths(exam_dir)["baseline"]
    have = {(r["start"], int(r["j"])) for r in fg.read_jsonl(path)}
    jd = judge_client.CpuJudge(str(judge_path))
    jd.warm()
    n = 0
    with open(path, "a", encoding="utf-8") as fh:
        for sid in sorted(starts):
            todo = [j for j in range(draws) if (sid, j) not in have]
            if not todo:
                continue
            doc = gl.load_start(str(exam_paths(exam_dir)["starts"]), sid)
            for j in todo:
                world = fg._fresh_world(doc["doc"], sid, j, NS)  # noqa: SLF001
                try:
                    row = jd.rows([world.handle])[0]
                    max_hp = world.max_hp()
                finally:
                    world.close()
                est = judge_client.estimate(row, max_hp)
                p_win = 1.0 - est.p_death
                l0 = float(judge_client.l0_of(p_win, est.hp_if_win / max(1, max_hp), est.potions_if_win, max_hp))
                fh.write(json.dumps({"start": sid, "j": j, "p_win": round(p_win, 5), "l0": round(l0, 5), "value": round(est.value, 3),
                                     "judge": Path(judge_path).name}) + "\n")
                n += 1
            fh.flush()
    say(exam_dir, f"baseline: {n} opening positions valued by the judge")
    return n


# ---------------------------------------------------------------------------------------------------------- the readings


def pct(x: float | None, d: int = 1) -> str:
    return "  -  " if x is None else f"{100 * x:.{d}f}"


def num(x: float | None, d: int = 2) -> str:
    return "  -  " if x is None else f"{x:.{d}f}"


def ci_text(b: dict, scale: float = 1.0, d: int = 2, unit: str = "") -> str:
    if b["estimate"] is None:
        return "n/a"
    lo, hi = b["lo"], b["hi"]
    rng = "" if lo is None or hi is None else f" [{scale * lo:+.{d}f}, {scale * hi:+.{d}f}]"
    return f"{scale * b['estimate']:+.{d}f}{unit}{rng}"


def cell_rows(games: dict, starts: dict, weights: dict | None = None) -> dict:
    """Per kind cell and per boss cell: starts, games, and the means of each measure over the starts with weight (all if None)."""
    sm = start_means(games, starts)
    out: dict[str, dict] = {}
    groups: dict[str, list[str]] = defaultdict(list)
    for sid in sm:
        if weights is not None and weights.get(sid, 0.0) <= 0.0:
            continue
        groups[starts[sid]["kind_cell"]].append(sid)
        if starts[sid]["kind"] == "boss":
            groups[starts[sid]["cell"]].append(sid)
    for c, ids in groups.items():
        out[c] = {"starts": len(ids), "games": sum(sm[s]["draws"] for s in ids),
                  **{m: mean(sm[s][m] for s in ids) for m in MEASURES}, "dmg_per_energy": mean(sm[s]["dmg_per_energy"] for s in ids)}
    return out


def cell_sort(c: str) -> tuple:
    kc = kind_of_cell(c)
    return (CELL_ORDER.index(kc) if kc in CELL_ORDER else 99, c.count("-"), c)


def format_cells(rows: dict, title: str) -> str:
    head = f"{'cell':<34}{'starts':>7}{'games':>7}{'win%':>7}{'L0':>7}{'hp lost':>9}{'potions':>8}{'turns':>7}{'dmg/en':>8}{'wasted blk':>11}"
    lines = [title, head, "-" * len(head)]
    for c in sorted(rows, key=cell_sort):
        r = rows[c]
        lines.append(f"{cell_label(c):<34}{r['starts']:>7}{r['games']:>7}{pct(r['won']):>7}{num(r['L0'], 3):>7}{num(r['hp_lost'], 1):>9}"
                     f"{num(r['potions_used']):>8}{num(r['turns'], 1):>7}{num(r['dmg_per_energy']):>8}{num(r['wasted'], 1):>11}")
    return "\n".join(lines)


def precision(games: dict, starts: dict, boot: int = 400) -> dict:
    """How well the exam measures one player: the win rate of each half over its starts with its bootstrap standard error (starts
    resampled within cells), the smallest difference two unpaired players would need to differ by to be told apart at 95% (1.96
    times the square root of two times the standard error), and the start-level agreement of draws 0-1 with draws 2-3 (Pearson r of a
    start's win rate on the two pairs, over starts with all four draws: the share of the variation that belongs to the start and not to
    the shuffles)."""
    out: dict = {}
    for h in "AB":
        groups: dict[str, list[dict]] = defaultdict(list)
        sm = start_means(games, {s: v for s, v in starts.items() if v["half"] == h})
        for sid, m in sm.items():
            groups[starts[sid]["cell"]].append({"won": m["won"]})
        b = bootstrap(groups, mean_stat("won"), boot=boot)
        out[h] = {"rate": b["estimate"], "se": b["se"], "starts": b["n"],
                  "unpaired_gap": None if b["se"] is None else 1.96 * math.sqrt(2.0) * b["se"]}
    ids = [s for s in starts if all((s, j) in games for j in range(DRAWS))]
    if len(ids) > 2:
        lo = [mean(game_values(games[(s, j)])["won"] for j in (0, 1)) for s in ids]
        hi = [mean(game_values(games[(s, j)])["won"] for j in (2, 3)) for s in ids]
        out["split_r"] = fg.pearson(lo, hi)
        out["split_starts"] = len(ids)
    return out


def precision_lines(games: dict, starts: dict) -> list[str]:
    p = precision(games, starts)
    lines = []
    for h in "AB":
        v = p[h]
        if v["rate"] is None:
            continue
        gap = "" if v["unpaired_gap"] is None else f"; two unpaired players would have to differ by {100 * v['unpaired_gap']:.1f} points to be told apart"
        lines.append(f"half {h}: win rate {100 * v['rate']:.1f}% over {v['starts']} starts, standard error {100 * (v['se'] or 0):.2f} points{gap}")
    if p.get("split_r") is not None:
        lines.append(f"start-level agreement of draws 0-1 with draws 2-3: Pearson r {p['split_r']:.2f} over {p['split_starts']} starts "
                     "(the closer to 1, the more of the variation is the start's, and the more a paired design gains)")
    return lines


def report_text(games: dict, starts: dict, label: str, cal: dict | None = None, baseline: dict | None = None) -> str:
    """The readings of one player: per-cell win rates and fight-quality measures (over the starts with weight, the calibration's
    always and never won starts counted out and listed), per-boss rates, the flags, the halves, the baseline's agreement."""
    cal = cal or calibration(games, starts)
    w = weights_of(cal, starts)
    played = [s for s in starts if any((s, j) in games for j in range(DRAWS))]
    half_n = Counter(starts[s]["half"] for s in starts)
    half_g = Counter(starts[s]["half"] for (s, j) in games if s in starts)
    out = [f"fight exam, player {label}: {len(played)} of {len(starts)} starts played, {sum(1 for (s, _j) in games if s in starts)} games "
           f"(half A {half_n['A']} starts / {half_g['A']} games, half B {half_n['B']} / {half_g['B']})"]
    out.append(format_cells(cell_rows(games, starts), "all starts (the calibration's flags not applied)"))
    out.append("")
    out.append(format_cells(cell_rows(games, starts, w), "starts won on some but not all draws (the weight-one set the statistics read)"))
    out.append("")
    flag = defaultdict(lambda: [0, 0, 0, 0])
    always, never, mixed = set(cal["always"]), set(cal["never"]), set(cal["mixed"])
    for sid in starts:
        k = starts[sid]["kind_cell"]
        flag[k][3] += 1
        if sid in always:
            flag[k][0] += 1
        elif sid in never:
            flag[k][1] += 1
        elif sid in mixed:
            flag[k][2] += 1
    out.append("calibration flags (weight zero: always won, never won)")
    out.append(f"{'cell':<20}{'always':>8}{'never':>8}{'mixed':>8}{'starts':>8}")
    for k in sorted(flag, key=cell_sort):
        a, n, m, t = flag[k]
        out.append(f"{k:<20}{a:>8}{n:>8}{m:>8}{t:>8}")
    out.append(f"{'all':<20}{len(cal['always']):>8}{len(cal['never']):>8}{len(cal['mixed']):>8}{len(starts):>8}"
               f"   (not yet fully played: {len(cal['unplayed'])})")
    pl = precision_lines(games, starts)
    if pl:
        out.append("")
        out.append("precision of the exam on this player")
        out.extend("  " + x for x in pl)
    if baseline:
        sm = start_means(games, starts)
        ids = [s for s in sm if s in baseline]
        xs, ys = [baseline[s]["p_win"] for s in ids], [sm[s]["won"] for s in ids]
        r = fg.pearson(xs, ys) if len(ids) > 2 else None
        out.append("")
        out.append(f"the judge's opening P(win) against the player's win rate over {len(ids)} starts: Pearson "
                   f"{'n/a' if r is None else f'{r:.2f}'}, mean P(win) {num(mean(xs), 3)}, mean win rate {num(mean(ys), 3)}")
    return "\n".join(out)


def compare_text(champ: dict, cand: dict, starts: dict, cal: dict, baseline: dict | None, champ_label: str, cand_label: str,
                 half: str = "A", boot: int = BOOT) -> str:
    """The candidate's exam against the champion's, on the draws both played of half `half`'s starts of weight one: the paired
    differences of each measure with a bootstrap interval, the baseline-adjusted win difference, the cross-fitted one, the
    McNemar on games and the per-cell wins. It decides nothing."""
    sub = {s: v for s, v in starts.items() if half == "all" or v["half"] == half}
    w = weights_of(cal, sub)
    groups = paired_starts(champ, cand, sub, w)
    n_starts = sum(len(v) for v in groups.values())
    games_n = sum(it["draws"] for v in groups.values() for it in v)
    mc = mcnemar_counts(champ, cand, sub, w)
    lines = [f"fight exam, {cand_label} against {champ_label} on half {half}: {n_starts} starts of weight one, {games_n} paired games "
             f"({sum(1 for s in sub if w.get(s, 0) <= 0)} of the half's {len(sub)} starts carry weight zero)"]
    lines.append(f"McNemar on the games: {cand_label}-only wins {mc['cand_only']}, {champ_label}-only {mc['champ_only']}, one-sided exact p "
                 f"{mc['p']:.3f}  (reported beside the promotion rule's, which decides)")
    b = bootstrap(groups, mean_stat("d_won"), boot=boot)
    lines.append(f"win rate difference, raw        : {ci_text(b, 100, 2, ' points')}  (bootstrap over starts, within cells)")
    if baseline:
        attach_baseline(groups, baseline)
        ref = reference_means(starts, weights_of(cal, starts), baseline)
        if ref is not None:
            adj = bootstrap(groups, adjusted_stat("d_won", ref), boot=boot)
            lines.append(f"win rate difference, adjusted   : {ci_text(adj, 100, 2, ' points')}  (to the exam's mean judge baseline; "
                         f"interval width {_width(adj, 100):.2f} against {_width(b, 100):.2f} raw)")
        else:
            lines.append("win rate difference, adjusted   : no baseline for these starts")
    else:
        lines.append("win rate difference, adjusted   : no baseline (run `fight_exam.py baseline`)")
    cf = crossfit_difference(champ, cand, sub, boot=boot)
    lines.append(f"win rate difference, cross-fitted: {ci_text(cf, 100, 2, ' points')}  (flags from two draws, difference on the other two)")
    for m, name, scale, d in (("d_L0", "L0 (0 to 1)", 1, 4), ("d_hp_lost", "HP lost a fight", 1, 2), ("d_potions_used", "potions spent", 1, 3),
                              ("d_turns", "turns", 1, 2), ("d_dmg_per_energy", "enemy HP removed per energy", 1, 3),
                              ("d_wasted", "block wasted a fight", 1, 2)):
        lines.append(f"{name:<32}: {ci_text(bootstrap(groups, mean_stat(m), boot=boot), scale, d)}")
    rows = []
    for c in sorted(groups, key=cell_sort):
        v = groups[c]
        rows.append((c, len(v), mean(it["champ_won"] for it in v), mean(it["cand_won"] for it in v), mean(it["d_hp_lost"] for it in v)))
    lines.append("")
    lines.append(f"{'cell':<34}{'starts':>7}{champ_label[:8]:>10}{cand_label[:8]:>10}{'diff pts':>10}{'hp lost diff':>14}")
    for c, n, a, b2, hp in rows:
        lines.append(f"{cell_label(c):<34}{n:>7}{pct(a):>10}{pct(b2):>10}{'  -  ' if a is None or b2 is None else f'{100 * (b2 - a):+.1f}':>10}"
                     f"{num(hp, 2):>14}")
    return "\n".join(lines)


def _width(b: dict, scale: float) -> float:
    return float("nan") if b["lo"] is None or b["hi"] is None else scale * (b["hi"] - b["lo"])


# ----------------------------------------------------------------------------------------------- the human line (readings)


def human_items(games: dict, human: dict) -> dict[str, list[dict]]:
    """The human stratum's starts by cell, each with the player's means over its draws (`p_...`), the human's own result (`h_...`) and the
    player's minus the human's (`d_...`), for survival (`won`), HP lost (a death loses the whole HP going in on both sides), the gym's
    L0 (0 to 1), turns and potions spent. Starts the player has no game of are left out."""
    sm = start_means(games, human)
    out: dict[str, list[dict]] = defaultdict(list)
    for sid, m in sm.items():
        s = human[sid]
        h_won = 1.0 if s.get("human_survived") else 0.0
        item = {"start": sid, "draws": m["draws"], "p_won": m["won"], "h_won": h_won, "d_won": m["won"] - h_won}
        for key, hk in (("hp_lost", "human_hp_lost"), ("L0", "human_l0_unit"), ("turns", "human_turns"), ("potions_used", "human_potions_used")):
            pv, hv = m[key], s.get(hk)
            item["p_" + key], item["h_" + key] = pv, hv
            item["d_" + key] = None if pv is None or hv is None else pv - hv
        out[s["cell"]].append(item)
    return out


def kind_of_cell(cell: str) -> str:
    k = "-".join(cell.split("-")[:2])
    return k.rstrip("0123456789") if k.startswith("act3-boss") else k  # act3-boss1-80 and act3-boss2-80 are one kind


def human_row(label: str, groups: dict, boot: int) -> dict:
    """One line of the human reading over `groups` {cell: [item]}: counts, the means of the two sides and the paired differences with
    their bootstrap intervals (survival and HP lost)."""
    items = [it for v in groups.values() for it in v]
    return {"label": label, "starts": len(items), "games": sum(it["draws"] for it in items),
            "p_won": mean(it["p_won"] for it in items), "h_won": mean(it["h_won"] for it in items),
            "p_hp_lost": mean(it["p_hp_lost"] for it in items), "h_hp_lost": mean(it["h_hp_lost"] for it in items),
            "p_L0": mean(it["p_L0"] for it in items), "h_L0": mean(it["h_L0"] for it in items),
            "d_won": bootstrap(groups, mean_stat("d_won"), boot=boot), "d_hp_lost": bootstrap(groups, mean_stat("d_hp_lost"), boot=boot)}


def human_rows(items: dict[str, list[dict]], boot: int = BOOT, detail: bool = True) -> list[dict]:
    """The lines: each act-and-kind cell (and under a boss kind each boss with `HUMAN_MIN_ROW` starts, when `detail`), then all."""
    rows = []
    kinds = sorted({kind_of_cell(c) for c in items}, key=lambda k: CELL_ORDER.index(k) if k in CELL_ORDER else 99)
    for k in kinds:
        groups = {c: v for c, v in items.items() if kind_of_cell(c) == k}
        rows.append({**human_row(k, groups, boot), "level": 0})
        if detail and k.endswith("boss"):
            rows += [{**human_row(c, {c: items[c]}, boot), "level": 1} for c in sorted(groups) if len(items[c]) >= HUMAN_MIN_ROW]
    rows.append({**human_row("all", items, boot), "level": 0})
    return rows


def human_table(rows: list[dict], title: str) -> str:
    head = (f"{'cell':<34}{'starts':>7}{'games':>7} | {'surv%':>6}{'human':>7}{'diff pts [95%]':>24} | {'HP lost':>8}{'human':>7}{'diff HP [95%]':>24} |"
            f"{'L0':>6}{'human':>7}")
    lines = [title, head, "-" * len(head)]
    for r in rows:
        name = ("  " * r["level"]) + cell_label(r["label"])
        lines.append(f"{name:<34}{r['starts']:>7}{r['games']:>7} | {pct(r['p_won']):>6}{pct(r['h_won']):>7}{ci_text(r['d_won'], 100, 1):>24} | "
                     f"{num(r['p_hp_lost'], 1):>8}{num(r['h_hp_lost'], 1):>7}{ci_text(r['d_hp_lost'], 1, 1):>24} |"
                     f"{num(r['p_L0'], 3):>6}{num(r['h_L0'], 3):>7}")
    return "\n".join(lines)


def human_report_text(games: dict, human: dict, label: str, boot: int = BOOT, detail: bool = True) -> str:
    """The human line of one player: his survival and HP lost against the human's own on the same starts (the human stratum),
    by act and kind and by boss, with the paired differences (player minus human). Positive survival and negative HP lost mean the
    player did better than the human."""
    played = [s for s in human if any((s, j) in games for j in range(DRAWS))]
    n_games = sum(1 for (s, _j) in games if s in human)
    half_n = Counter(human[s]["half"] for s in played)
    head = (f"fight exam, human stratum, player {label}: {len(played)} of {len(human)} starts played, {n_games} games "
            f"(half A {half_n['A']} starts, half B {half_n['B']}); human survived on {sum(1 for s in played if human[s].get('human_survived'))} "
            f"of the played starts")
    if not played:
        return head
    items = human_items(games, human)
    table = human_table(human_rows(items, boot, detail), "player minus human, paired on the same starts (survival in points, HP lost in HP; "
                        "a death costs the whole HP going in on both sides)")
    return head + "\n" + table


def human_compare_text(champ: dict, cand: dict, human: dict, champ_label: str, cand_label: str, half: str = "A", boot: int = BOOT) -> str:
    """The human line in a comparison: the candidate against the human (full table), the champion against the human (the same starts,
    act and kind lines), and the candidate against the champion on those starts, paired (no calibration flags: the human is not a draw)."""
    sub = {s: v for s, v in human.items() if half == "all" or v["half"] == half}
    if not sub:
        return "human stratum: none in the exam"
    out = [f"human stratum, half {half}: {len(sub)} starts"]
    out.append(human_report_text(cand, sub, cand_label, boot))
    if any(s in sub for (s, _j) in champ):
        out += ["", human_report_text(champ, sub, champ_label, boot, detail=False)]
    w = {sid: 1.0 for sid in sub}
    groups = paired_starts(champ, cand, sub, w)
    n_starts = sum(len(v) for v in groups.values())
    if n_starts:
        out += ["", f"{cand_label} minus {champ_label} on the human stratum: {n_starts} starts, {sum(it['draws'] for v in groups.values() for it in v)} paired games"]
        out.append(f"{'win rate difference':<32}: {ci_text(bootstrap(groups, mean_stat('d_won'), boot=boot), 100, 2, ' points')}")
        out.append(f"{'HP lost a fight':<32}: {ci_text(bootstrap(groups, mean_stat('d_hp_lost'), boot=boot), 1, 2)}")
        out.append(f"{'L0 (0 to 1)':<32}: {ci_text(bootstrap(groups, mean_stat('d_L0'), boot=boot), 1, 4)}")
        kinds: dict[str, dict] = defaultdict(dict)
        for c, v in groups.items():
            kinds[kind_of_cell(c)][c] = v
        for k in sorted(kinds, key=lambda k: CELL_ORDER.index(k) if k in CELL_ORDER else 99):
            b, h = bootstrap(kinds[k], mean_stat("d_won"), boot=boot), bootstrap(kinds[k], mean_stat("d_hp_lost"), boot=boot)
            out.append(f"  {k:<30}: win {ci_text(b, 100, 1, ' pts')} over {b['n']} starts; HP lost {ci_text(h, 1, 1)}")
    else:
        out.append(f"\nno start of the human stratum has games of both {cand_label} and {champ_label} yet")
    return "\n".join(out)


# ------------------------------------------------------------------------------------------ the human bar at a judgement (report-only)


def human_games(rows: list[dict], human: dict, config: str | None = None) -> dict:
    """{(start, j): row} of the rows that are games of the human stratum, and with `config` only those played under it."""
    return index_games([r for r in rows if r.get("start") in human and (config is None or r.get("config", config) == config)])


def _tags(games: dict, field: str) -> list[str]:
    return sorted({str(r.get(field)) for r in games.values() if r.get(field)})


def _r(x: float | None, d: int = 2) -> float | None:
    return None if x is None else round(x, d)


def _single_groups(games: dict, human: dict) -> dict:
    """{cell: [item]} of one player against the human (`human_items`), with the other side's keys empty."""
    return {c: [{**it, "q_won": None, "q_hp_lost": None} for it in v] for c, v in human_items(games, human).items()}


def human_bar_reading(cand: dict, inc: dict | None, human: dict, boot: int = BOOT, draws: int = DRAWS) -> dict:
    """The human line of one judgement from the games (`{(start, j): row}`) of a candidate and, if it has any, an incumbent, on the
    human stratum `human`: survival (in points per 100) and HP lost of the candidate, the incumbent and the human on the same starts
    (those both played; the candidate's alone when the incumbent has no game), and the candidate minus the incumbent on them, paired
    by start on the draws both have (the mean over the draws), with its 95% bootstrap interval over starts (within cells). A death
    costs the whole HP going in on both sides. Report-only: nothing here reads or changes a promotion rule."""
    cand = {k: r for k, r in cand.items() if k[0] in human}
    inc = {k: r for k, r in (inc or {}).items() if k[0] in human}
    out: dict = {"rule": HUMAN_RULE, "stratum_starts": len(human), "cand_games": len(cand), "inc_games": len(inc),
                 "cand_engine": _tags(cand, "engine"), "inc_engine": _tags(inc, "engine") if inc else None,
                 "complete": bool(human) and all((sid, j) in cand for sid in human for j in range(draws))}
    out["same_engine"] = (out["cand_engine"] == out["inc_engine"]) if inc else None
    if not cand:
        return {**out, "starts": 0}
    groups: dict = {}
    if inc:
        for c, v in paired_starts(inc, cand, human, {sid: 1.0 for sid in human}).items():
            groups[c] = [{**it, "p_won": it["cand_won"], "q_won": it["champ_won"], "p_hp_lost": it["cand_hp_lost"],
                          "q_hp_lost": it["champ_hp_lost"], "h_won": 1.0 if human[it["start"]].get("human_survived") else 0.0,
                          "h_hp_lost": human[it["start"]].get("human_hp_lost")} for it in v]
    if not groups:  # no incumbent game, or none on a start of the candidate's: the candidate's side alone
        groups = _single_groups(cand, human)
        out.update(inc_games=0, inc_engine=None, same_engine=None)
        inc = {}
    items = [it for v in groups.values() for it in v]

    def m(key: str) -> float | None:
        return mean(it.get(key) for it in items)

    out.update(starts=len(items), cand_surv=_r(100 * m("p_won")), human_surv=_r(100 * m("h_won")), cand_hp_lost=_r(m("p_hp_lost")),
               human_hp_lost=_r(m("h_hp_lost")))
    if inc:
        b, h = bootstrap(groups, mean_stat("d_won"), boot=boot), bootstrap(groups, mean_stat("d_hp_lost"), boot=boot)

        def pts(x: float | None) -> float | None:
            return None if x is None else _r(100 * x)

        out.update(inc_surv=_r(100 * m("q_won")), inc_hp_lost=_r(m("q_hp_lost")), diff_starts=len(items),
                   diff_surv=pts(b["estimate"]), diff_surv_lo=pts(b["lo"]), diff_surv_hi=pts(b["hi"]),
                   diff_hp_lost=_r(h["estimate"]), diff_hp_lost_lo=_r(h["lo"]), diff_hp_lost_hi=_r(h["hi"]))
    return out


def human_bar_line(r: dict, inc: str | None = None) -> str:
    """The reading as one line, e.g. `human bar: cand 55.0 / inc 62.0 / human 79.8 in 100 on 400 starts; cand-inc -7.0 [-11.2, -2.8];
    HP lost 41.2 / 38.6 / 33.2`. `inc` names the incumbent for the case it has no game."""
    if not r.get("starts"):
        return "human bar: no human-stratum games of the candidate"
    n = f"{r['starts']} starts" if r["starts"] == r["stratum_starts"] else f"{r['starts']} of {r['stratum_starts']} starts"
    if r.get("inc_surv") is None:
        who = f" {inc}" if inc else ""
        return (f"human bar: cand {r['cand_surv']:.1f} / human {r['human_surv']:.1f} in 100 on {n} (no human-stratum games for the "
                f"incumbent{who}); HP lost {r['cand_hp_lost']:.1f} / {r['human_hp_lost']:.1f}")
    d = r.get("diff_surv")
    diff = "n/a" if d is None else f"{d:+.1f}" + ("" if r.get("diff_surv_lo") is None else f" [{r['diff_surv_lo']:+.1f}, {r['diff_surv_hi']:+.1f}]")
    line = (f"human bar: cand {r['cand_surv']:.1f} / inc {r['inc_surv']:.1f} / human {r['human_surv']:.1f} in 100 on {n}; cand-inc {diff}; "
            f"HP lost {r['cand_hp_lost']:.1f} / {r['inc_hp_lost']:.1f} / {r['human_hp_lost']:.1f}")
    if r.get("same_engine") is False:
        line += f"; engines differ (cand {','.join(r['cand_engine'])}, inc {','.join(r['inc_engine'])})"
    return line


def games_date(path: Path) -> str | None:
    """The UTC date a games file was last written (the rows carry no time)."""
    try:
        return dt.datetime.fromtimestamp(Path(path).stat().st_mtime, tz=dt.timezone.utc).strftime("%Y-%m-%d")
    except OSError:
        return None


def gym_player_names(exam_dir: Path = EXAM_DIR, loop_root: Path = LOOP_ROOT) -> list[tuple[str, Path]]:
    """The gym players as (name, folder): every promoted version under `players/` (newest first), then the checkpoints that have a
    games file in the exam folder."""
    root = Path(loop_root)
    versions = [p for p in (root / "players").glob("v*") if p.name[1:].isdigit() and (p / "combat_policy.pt").is_file()]
    out = [(p.name, p) for p in sorted(versions, key=lambda p: -int(p.name[1:]))]
    for g in sorted(Path(exam_dir).glob("games-*.jsonl")):
        name = g.name[len("games-"):-len(".jsonl")]
        d = root / "checkpoints" / name
        if (d / "combat_policy.pt").is_file() and all(name != n for n, _ in out):
            out.append((name, d))
    return out


def gym_player_of(policy_sha: str, exam_dir: Path = EXAM_DIR, loop_root: Path = LOOP_ROOT) -> str | None:
    """The name of a gym player whose `combat_policy.pt` is the network with this sha256, among the players with games in the exam
    folder (None: no such player). A run-level agent's fight network is a gym player's when the two files are the same bytes."""
    for name, d in gym_player_names(exam_dir, loop_root):
        if gl.policy_sha(d) == policy_sha:
            return name
    return None


def lab_human_bar(policy_sha: str | None, exam_dir: Path | None = None, loop_root: Path | None = None) -> dict:
    """The one report-only line of a lab judgement (the keep test v2): the judged player's fight network (the sha256 of its
    `combat_policy.pt`, named by 12 hex digits and by the gym player that holds the same bytes) and that player's latest reading on the
    human stratum from the exam's games: survival and HP lost of the player and of the human on the same starts, and the games' date.
    No game is played. A network no gym player holds gets the gym incumbent's reading, labelled as the incumbent's. Returns {line, ...}."""
    exam_dir, loop_root = Path(exam_dir or EXAM_DIR), Path(loop_root or LOOP_ROOT)
    if not policy_sha:
        return {"line": "human bar: the judged player has no fight network (no combat prior), nothing to read"}
    net = policy_sha[:12]
    human = load_human_exam(exam_dir)
    if not human:
        return {"line": f"human bar: no human-stratum games for fight network {net} (the exam has no human stratum)", "network": net}
    name = gym_player_of(policy_sha, exam_dir, loop_root)
    held = name is not None
    who = name if held else champion_label(loop_root)
    path = games_path(exam_dir, who)
    games = human_games(fg.read_jsonl(path), human, DEFAULT_CONFIG) if path.is_file() else {}
    desc = f"fight network {net}" + (f" (gym player {name})" if held else " (no gym player holds these bytes)")
    if not games:
        extra = "" if held else f"; the gym incumbent {who} has none either"
        return {"line": f"human bar: no human-stratum games for {desc}{extra}", "network": net, "gym_player": name, "reading_of": who}
    r = human_bar_reading(games, None, human, boot=0)
    date = games_date(path)
    nums = (f"survival {r['cand_surv']:.1f} / human {r['human_surv']:.1f} in 100, HP lost {r['cand_hp_lost']:.1f} / {r['human_hp_lost']:.1f} "
            f"on {r['starts']} of {r['stratum_starts']} starts, {r['cand_games']} games, engine {','.join(r['cand_engine']) or 'unknown'}, "
            f"games file written {date or 'unknown date'}")
    line = f"human bar: {desc}: {nums}" if held else f"human bar: {desc}; the gym incumbent {who}'s reading, not this network's: {nums}"
    return {"line": line, "network": net, "gym_player": name, "reading_of": who, "date": date,
            **{k: r[k] for k in ("starts", "cand_games", "cand_surv", "human_surv", "cand_hp_lost", "human_hp_lost", "cand_engine")}}


def load_baseline(d: Path) -> dict | None:
    rows = fg.read_jsonl(exam_paths(d)["baseline"])
    return baseline_by_start(rows) if rows else None


def calibration_path(d: Path, label: str) -> Path:
    return Path(d) / f"calibration-{label}.json"


def games_path(d: Path, label: str) -> Path:
    return Path(d) / f"games-{label}.jsonl"


def champion_label(loop_root: Path = LOOP_ROOT) -> str:
    return (gl.read_json(Path(loop_root) / "state.json", {}) or {}).get("incumbent", "v2")


def cmd_report(a: argparse.Namespace) -> int:
    d = Path(a.exam_dir)
    starts = load_exam(d)
    label = a.player or champion_label()
    all_games = index_games(fg.read_jsonl(games_path(d, label)))
    games = {k: v for k, v in all_games.items() if k[0] in starts}  # the human stratum's rows are read below, not here
    human = load_human_exam(d)
    human_games = {k: v for k, v in all_games.items() if k[0] in human}
    if not games and not human_games:
        print(f"no games of {label} in {games_path(d, label).name}")
        return 1
    if not games:
        print(human_report_text(human_games, human, label, boot=getattr(a, "boot", BOOT)))
        return 0
    cal = calibration(games, starts)
    if a.write_calibration:
        gl.write_json(calibration_path(d, label), {**cal, "player": label, "written": gl.stamp(), "rule": EXAM_RULE})
    meta = gl.read_json(exam_paths(d)["meta"], {}) or {}
    print(report_text(games, starts, label, cal, load_baseline(d)))
    short = meta.get("short_cells") or []
    if short:
        print("\ncells short of the target " + str(meta.get("per_cell")) + ": " + ", ".join(
            f"{cell_label(c)} {meta['by_cell'][c]['chosen_total']}" for c in short))
    if human_games:
        print("\n" + human_report_text(human_games, human, label, boot=getattr(a, "boot", BOOT)))
    return 0


def cmd_compare(a: argparse.Namespace) -> int:
    d = Path(a.exam_dir)
    starts = load_exam(d)
    champ = a.champion or champion_label()
    cand_label = Path(a.candidate).name
    champ_all = index_games(fg.read_jsonl(games_path(d, champ)))
    champ_games = {k: v for k, v in champ_all.items() if k[0] in starts}
    human = load_human_exam(d)
    cal = gl.read_json(calibration_path(d, champ), None)
    if cal is None:
        if not champ_games:
            print(f"no calibration games of {champ}: run `fight_exam.py play --player {champ}` first")
            return 1
        cal = calibration(champ_games, starts)
        print("note: no calibration file for the champion; flags taken from its games as they stand")
    if not a.no_play:
        cand_dir = resolve_player(a.candidate)
        rc = run_player(a, cand_label, cand_dir, "A", games_path(d, cand_label))
        if rc:
            return rc
    cand_all = index_games(fg.read_jsonl(games_path(d, cand_label)))
    cand_games = {k: v for k, v in cand_all.items() if k[0] in starts}
    if not cand_games:
        print(f"no games of {cand_label}")
        return 1
    for field in ("config", "prior_path", "engine"):
        seen = {r.get(field) for r in list(champ_games.values())[:50]} | {r.get(field) for r in list(cand_games.values())[:50]}
        seen.discard(None)
        if len(seen) > 1:
            print(f"WARNING: the two players' games differ in {field}: {sorted(map(str, seen))}")
    print(compare_text(champ_games, cand_games, starts, cal, load_baseline(d), champ, cand_label, half="A", boot=a.boot))
    if human and any(k[0] in human for k in cand_all):
        print("\n" + human_compare_text({k: v for k, v in champ_all.items() if k[0] in human}, {k: v for k, v in cand_all.items() if k[0] in human},
                                         human, champ, cand_label, half="A", boot=a.boot))
    return 0


def cmd_select(a: argparse.Namespace) -> int:
    """Build the exam. The loop is paused while it runs (the lock): its next step would mark more starts as practised in state.json,
    and `clean` is read from there."""
    kw = dict(per_cell=a.per_cell, extra_path=Path(a.extra_starts) if a.extra_starts else None, allow_practised=a.allow_practised,
              rebuild=a.rebuild)
    if a.no_lock:
        meta = build_exam(LOOP_ROOT, Path(a.exam_dir), **kw)
    else:
        try:
            with ExamLock(until=None, stop_file=Path(a.exam_dir) / "STOP"):
                meta = build_exam(LOOP_ROOT, Path(a.exam_dir), **kw)
        except Blocked as why:
            print(f"not started: {why}")
            return 3
    print(json.dumps({k: meta[k] for k in ("starts", "runs", "halves", "by_kind_cell", "short_cells", "practised_starts", "by_stratum")}, indent=1))
    print("cell                              pool  clean  chosen")
    for c in sorted(meta["by_cell"], key=cell_sort):
        v = meta["by_cell"][c]
        print(f"{cell_label(c):<32}{v.get('pool', 0):>6}{v.get('clean', 0):>7}{v.get('chosen_total', 0):>8}{'  SHORT' if v.get('short') else ''}")
    return 0


def cmd_add_human(a: argparse.Namespace) -> int:
    """Add the human stratum. The loop is paused while it runs (the lock) so the plays it is picked against are the plays there are."""
    kw = dict(per_cell=a.per_cell, total=a.total, rebuild=a.rebuild, starts_dir=Path(a.starts_dir) if a.starts_dir else None)
    try:
        if a.no_lock:
            meta = build_human(LOOP_ROOT, Path(a.exam_dir), **kw)
        else:
            with ExamLock(until=None, stop_file=Path(a.exam_dir) / "STOP"):
                meta = build_human(LOOP_ROOT, Path(a.exam_dir), **kw)
    except Blocked as why:
        print(f"not started: {why}")
        return 3
    print(json.dumps({k: meta[k] for k in ("starts", "runs", "halves", "by_kind_cell", "short_cells", "left_out", "played_runs", "survived")}, indent=1))
    print("cell                              pool   runs  quota  chosen")
    for c in sorted(meta["by_cell"], key=cell_sort):
        v = meta["by_cell"][c]
        print(f"{cell_label(c):<32}{v['pool']:>6}{v['runs']:>7}{v['quota']:>7}{v['chosen_total']:>8}{'  SHORT' if v['short'] else ''}")
    return 0


def cmd_add_act3(a: argparse.Namespace) -> int:
    """Add the Act 3 stratum. The loop is paused while it runs (the lock), as for the human stratum."""
    kw = dict(starts_dir=Path(a.starts_dir), rebuild=a.rebuild, stand_alone=a.stand_alone)
    try:
        if a.no_lock:
            meta = build_act3(Path(a.exam_dir), **kw)
        else:
            with ExamLock(until=None, stop_file=Path(a.exam_dir) / "STOP"):
                meta = build_act3(Path(a.exam_dir), **kw)
    except Blocked as why:
        print(f"not started: {why}")
        return 3
    print(json.dumps({k: meta[k] for k in ("starts", "runs", "halves", "by_stratum", "left_out", "human_survived", "practised_starts")}, indent=1))
    print("cell                              chosen   human: pool clean quota chosen clean    own: pool quota chosen")
    for c, v in meta["by_cell"].items():
        h, o = v["human"], v["own"]
        print(f"{cell_label(c):<32}{v['chosen_total']:>7}   {h.get('pool', 0):>16}{h.get('clean_pool', 0):>6}{h.get('quota', 0):>6}{h.get('chosen', 0):>7}"
              f"{h.get('clean', 0):>6}   {o.get('pool', 0):>10}{o.get('quota', 0):>6}{o.get('chosen', 0):>7}{'  SHORT' if h.get('short') or o.get('short') else ''}")
    return 0


def cmd_baseline(a: argparse.Namespace) -> int:
    d = Path(a.exam_dir)
    starts = load_exam(d)
    if a.limit:
        starts = dict(sorted(starts.items(), key=lambda kv: order_key(SELECT_SEED, kv[0]))[: a.limit])
    prepare_process()
    try:
        with ExamLock(until=a.until or None, stop_file=d / "STOP"):
            n = make_baseline(d, a.judge_path, starts, draws=a.draws)
    except Blocked as why:
        print(f"not started: {why}")
        return 3
    print(f"baseline: {n} positions")
    return 0


def cmd_play(a: argparse.Namespace) -> int:
    d = Path(a.exam_dir)
    who = getattr(a, "player_dir", "") or a.player  # a folder (a checkpoint the gate is scoring) or a name under players/, checkpoints/
    if not who:
        print("play: --player or --player-dir is needed")
        return 2
    label = a.label or Path(who).name
    return run_player(a, label, resolve_player(who), a.half, games_path(d, label))


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p, play: bool = False):
        p.add_argument("--exam-dir", default=str(EXAM_DIR))
        p.add_argument("--allow-quiet", action="append", default=[], metavar="TAG",
                       help="a QUIET.lock tag this run does not wait for (repeatable; e.g. the lab's daytime score)")
        if play:
            p.add_argument("--stratum", choices=("all", "ai", "human"), default="all", help="which starts to play (the human stratum is a second file)")
            p.add_argument("--config", default=DEFAULT_CONFIG, help="the search config the exam plays (the gate's is the daemon's)")
            p.add_argument("--judge", default="", help="the judge checkpoint a judge config's leaves read")
            p.add_argument("--judge-path", default=str(gl.REF / "agent" / "judge" / "v0" / "judge.pt"), help="the baseline's judge")
            p.add_argument("--procs", type=int, default=6)
            p.add_argument("--k", type=int, default=32)
            p.add_argument("--prior", choices=("server", "cpu"), default="server")
            p.add_argument("--until", default="", help="stop at this UTC stamp (2026-11-04T15:45:00Z)")
            p.add_argument("--no-baseline", action="store_true")
            p.add_argument("--draws", type=int, default=DRAWS, help="draws a start (a smoke test plays fewer)")
            p.add_argument("--limit", type=int, default=0, help="play only this many starts, spread over the cells (a smoke test)")

    p = sub.add_parser("select")
    common(p)
    p.add_argument("--per-cell", type=int, default=PER_CELL)
    p.add_argument("--extra-starts", default="", help="a second start file in the gym's record schema (the human stratum)")
    p.add_argument("--allow-practised", action="store_true", help="top a short cell up with the least-played starts (marked practised)")
    p.add_argument("--rebuild", action="store_true", help="replace an existing exam (its files get a dated backup name)")
    p.add_argument("--no-lock", action="store_true", help="do not pause the loop (a scratch folder, the tests)")
    p = sub.add_parser("add-human")
    common(p)
    p.add_argument("--per-cell", type=int, default=HUMAN_PER_CELL)
    p.add_argument("--total", type=int, default=HUMAN_TOTAL)
    p.add_argument("--starts-dir", default="", help="the folder of human-starts.* and human-outcomes.json (default: the loop's starts/)")
    p.add_argument("--rebuild", action="store_true", help="replace an existing stratum (its files get a dated backup name)")
    p.add_argument("--no-lock", action="store_true", help="do not pause the loop (a scratch folder, the tests)")
    p = sub.add_parser("add-act3", help="add the Act 3 boss stratum (human openings and our own arrivals) as a third file")
    common(p)
    p.add_argument("--starts-dir", required=True, help="the folder of human-starts-act3.*, human-outcomes-act3.json and own-starts-act3.* (rl/act3_starts.py)")
    p.add_argument("--rebuild", action="store_true", help="replace an existing stratum (its files get a dated backup name)")
    p.add_argument("--stand-alone", action="store_true", help="the folder holds no AI exam (a staging folder)")
    p.add_argument("--no-lock", action="store_true", help="do not pause the loop (a scratch folder, the tests)")
    p = sub.add_parser("baseline")
    common(p, True)
    p = sub.add_parser("play")
    common(p, True)
    p.add_argument("--player", default="", help="a name under the loop's players/ then checkpoints/, or a folder")
    p.add_argument("--player-dir", default="", help="a player's folder (a checkpoint the gate scores); its games go to games-<folder name>.jsonl")
    p.add_argument("--label", default="")
    p.add_argument("--half", choices=("A", "B", "all"), default="all")
    p = sub.add_parser("report")
    common(p)
    p.add_argument("--boot", type=int, default=BOOT, help="bootstrap draws of the human line")
    p.add_argument("--player", default="")
    p.add_argument("--write-calibration", action="store_true", help="store the player's always/never-won flags (the champion's)")
    p = sub.add_parser("compare")
    common(p, True)
    p.add_argument("candidate")
    p.add_argument("--champion", default="")
    p.add_argument("--no-play", action="store_true")
    p.add_argument("--boot", type=int, default=BOOT)
    a = ap.parse_args(argv)
    ALLOW_QUIET[:] = list(getattr(a, "allow_quiet", None) or [])
    return {"select": cmd_select, "add-human": cmd_add_human, "add-act3": cmd_add_act3, "baseline": cmd_baseline, "play": cmd_play, "report": cmd_report,
            "compare": cmd_compare}[a.cmd](a)


if __name__ == "__main__":
    raise SystemExit(main())
