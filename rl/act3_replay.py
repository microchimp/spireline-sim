#!/usr/bin/env python3
"""What more search buys at the Act 3 bosses (the Act 3 week, lane 4; option b of the 2026-11-18 learning analysis).

    python rl/act3_replay.py positions --out DIR [--stores S ...] [--n-first 30] [--n-second 8] [--engine old|core2]
    python rl/act3_replay.py play      --out DIR [--agent-dir D] [--mults 1 4 16] [--procs 4] [--limit-pos N] [--n-per-mult 1:30,4:30,16:10]
    python rl/act3_replay.py report    --out DIR [--json]
    python rl/act3_replay.py size      --out DIR [--core-hours 90] [--procs 12] [--wall-hours 8]

The question. If the champion's fight player thinks four or sixteen times longer at an Act 3 boss, how often does it win the fight
that it lost or won with today's budget? "1x" is the champion's own boss budget (its agent.json `search_schedule`: today
`a3-v1@h1e4b16`, the boss multiplier 16 on the base config's 30 simulations a plain play and 120 at a target or a choice screen);
"4x" and "16x" are 4 and 16 times that (boss multiplier 64 and 256). Nothing else changes: the champion's fight network, the base
config, the same shuffles and per-decision search seeds at every budget (common random numbers, `fight_gym.game_seed`).

The positions. A plain run of the data step (`referee/run-cheap/<generation>/runs.jsonl`) writes one record per run
(`decisions.d/<seed>.record.json`) with, for every fight, the run state going in: deck (card id, upgraded, enchantment, amount),
relics, potions, HP, max HP, the relics' counters and the encounter. From that the copy's arena entry (`native.reset_arena`, the way
`rl/human_starts.py` rebuilds a human fight) deals the boss fight's opening: the same deck, relics, potions, HP and boss as the run
walked in with, at the floor's number, Ascension 10. Position = one Act 3 boss fight of one run: ordinal 1 is the first Act 3 boss,
2 the second (reached only by a run that beat the first). The sample is a deterministic hash order over the runs of the stores
(`SEED`), so the same stores give the same positions; the run's own result at the fight (`run_won`) is kept for the split.
What the rebuild cannot say: a card's bonuses gained during earlier fights (none persist in this game but enchantments that grow in
play), and a relic counter the run holds but the copy's relic does not read as a counter. Relic counters of the record are written
into the opening's relics; the count of positions that carry one is in the positions' meta.

The games. A position is played at each budget by `fight_gym.play_steps` (the gym's and the exam's own game loop) from the opening,
the hidden information drawn afresh from the game seeds of (position, draw) and never from the budget, on the champion's fight
prior in-process (the CPU path: no server, so the run is exact and independent of the GPU). One JSONL row per (position, budget,
draw): kill, HP left, turns, decisions, simulations, wall seconds. A game past its deadline or `fight_gym.DECISION_CAP` is capped,
counted as a loss in the headline and shown on its own line. Rows are appended as they finish and a game with a row is never played
twice, so a stop (the night, HALT, PAUSE, another's QUIET.lock line, `--until`, STOP) loses only the games in flight.

The report. Kills per budget with a 95% Wilson interval; the paired differences between budgets on the positions that have both
(a bootstrap over positions, and the sign test on the discordant pairs); the splits by boss and by the run's own result; cost per
budget (mean and median wall seconds, simulations); and `size`, the full run's size from a measured sample.

Lab code: nothing here changes the engine, the search or the solver. Heavy work belongs to the caller's QUIET.lock line
(`ACT3-REPLAY-full.sh` holds it); this module only stops when the machine is wanted.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import random
import statistics
import sys
import time
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Nothing here loads the emulator library at import: the pure parts run anywhere, the games in spawned processes.

RULE = "act3-replay-v1"
SEED = 20261119  # the sample's hash salt
NS = "r3"  # the draws' seed namespace (the gym's "g", the exam's "x", the practice's "L")
TAG = "act3-replay"  # the run's QUIET.lock line (held by ACT3-REPLAY-full.sh)
DEFAULT_STORES = ("D:/spireline/referee/run-cheap/17", "D:/spireline/referee/run-cheap/18")
DEFAULT_OUT = "D:/spireline/referee/backlog/act3-week/act3-replay-full"
DEFAULT_AGENT_CURRENT = "D:/spireline/referee/agent/current.json"
BASE_CONFIG = "a3-v1"
MULTS = (1, 4, 16)  # of the champion's own boss budget
ASCENSION = 10
NODE_BOSS = 6
ACT3_INDEX = 2  # run state's act_index of Act 3
DEADLINE_S = {1: 1800.0, 4: 5400.0, 16: 14400.0}  # a game's soft deadline by budget (the smoke's longest games: 43 s, 185 s, 654 s); the parent kills 1.5 times past it plus a minute
BOOT = 2000
BOOT_SEED = 20261120
FILES = {"positions": "positions.jsonl", "meta": "meta.json", "games": "games.jsonl", "log": "act3-replay.log"}


# ---------------------------------------------------------------------------------------------------------- pure parts


def arm_label(mult: int) -> str:
    return f"{int(mult)}x"


def order_key(salt: int, ordinal: int, gen, seed: str) -> str:
    """The deterministic draw order of a candidate run: a hash of the salt, the boss ordinal, the generation and the run's seed."""
    return hashlib.sha256(f"act3-replay:{salt}|{ordinal}|{gen}|{seed}".encode()).hexdigest()


def game_seed_of(pid: str, what: str) -> int:
    h = hashlib.sha256(f"{pid}|{what}".encode()).digest()
    return int.from_bytes(h[:4], "little") % (1 << 30)


def boss_name(encounter_id) -> str:
    try:
        from run_train import vocab  # noqa: PLC0415

        return vocab.BOSS_SLUGS.get(int(encounter_id), f"enc{encounter_id}")
    except (ImportError, ValueError, TypeError):
        return f"enc{encounter_id}"


def arm_config(stakes: str | None, hold: int | None, mult: int, base: str = BASE_CONFIG):
    """The search config of the arm that thinks `mult` times as long as the champion at a boss: the champion's own config
    (`search_configs.for_agent`: its stakes schedule and hold) with its boss multiplier times `mult`. Mult 1 is the champion's
    config itself. The hallway and elite multipliers are untouched (the replay plays boss fights only). The multiplier is set on
    the config's `stakes` field directly: `macro_recipe` caps a named schedule at 64, and the 16x arm's boss multiplier is 256."""
    import search_configs  # noqa: PLC0415 -- pure

    champion = search_configs.for_agent(base, stakes, hold)
    if int(mult) == 1:
        return champion
    hall, elite, boss = champion.stakes if champion.stakes is not None else (1, 1, 1)
    top = int(boss) * int(mult)
    return dataclasses.replace(champion, name=f"{base}@h{hall}e{elite}b{top}" + (f"+hold{hold}" if hold is not None else ""),
                               stakes=(hall, elite, top))


def position_spec(record: dict, ordinal: int, gen, store: str = "") -> dict | None:
    """The Act 3 boss fight `ordinal` (1 first, 2 second) of a decision record as a position spec, or None when the record does
    not hold it (the run did not get there, or the fight's state is not in it)."""
    fights = [f for f in record.get("fights") or [] if int(f.get("node_type", 0)) == NODE_BOSS
              and int((f.get("state") or {}).get("act_index", -1)) == ACT3_INDEX]
    if not 1 <= int(ordinal) <= len(fights):
        return None
    f = fights[int(ordinal) - 1]
    st = f.get("state") or {}
    if not st.get("deck") or not st.get("max_hp") or int(st.get("hp", 0)) <= 0:
        return None
    seed = str(record.get("seed"))
    counters = [[int(a), int(b)] for a, b in ((st.get("gaps") or {}).get("counters") or [])]
    return {"id": f"{gen}:{seed}|b{int(ordinal)}", "gen": gen, "seed": seed, "ordinal": int(ordinal), "floor": int(f["floor"]),
            "node_type": NODE_BOSS, "encounter_id": int(f["encounter"]), "boss": boss_name(f["encounter"]),
            "hp": int(st["hp"]), "max_hp": int(st["max_hp"]), "gold": int(st.get("gold", 0)),
            "deck": [[int(c[0]), int(c[1]), int(c[2]), int(c[3])] for c in st["deck"]],
            "relics": [int(x) for x in st.get("relics") or []], "potions": [int(x) for x in st.get("potions") or []],
            "potion_slots": int(st.get("max_potion_slots", 2)), "counters": counters,
            "run_won": bool(f.get("won")), "run_hp_after": int(f.get("hp_after", 0)), "store": store}


def reset_args(spec: dict) -> dict:
    """`native.reset_arena`'s arguments for a spec: negative card ids are upgraded, the enchantments run parallel to the deck."""
    ids, ench = [], []
    for card_id, up, e_id, e_amt in spec["deck"]:
        ids.append(-int(card_id) if int(up) >= 1 else int(card_id))
        ench.append((int(e_id), int(e_amt) if int(e_id) else 0))
    pots = list(spec["potions"])
    return {"deck_ids": ids, "enchantments": ench, "encounter_id": int(spec["encounter_id"]), "relic_ids": list(spec["relics"]),
            "potion_ids": pots, "player_hp": int(spec["hp"]), "player_max_hp": int(spec["max_hp"]), "player_gold": int(spec["gold"]),
            "ascension": ASCENSION, "total_floor": int(spec["floor"]), "max_potion_slots": max(int(spec["potion_slots"]), len(pots)),
            "completed_combat_rooms": -1}


def apply_counters(doc: dict, counters: list) -> int:
    """Write the run's relic counters into the opening's relics (same relic id); returns how many were written."""
    want = {int(a): int(b) for a, b in counters}
    n = 0
    for r in doc.get("relics") or []:
        if int(r.get("id", -1)) in want and want[int(r["id"])] != int(r.get("counter", 0)):
            r["counter"] = want[int(r["id"])]
            n += 1
    return n


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    """The Wilson score interval of k of n (a proportion); (0, 1) for n = 0."""
    if n <= 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def sign_test_p(b: int, c: int) -> float:
    """Two-sided exact sign test p of b wins against c losses among the discordant pairs (1.0 when there are none)."""
    n = b + c
    if n == 0:
        return 1.0
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(k + 1)) / 2**n
    return min(1.0, 2 * tail)


def percentile(xs: list[float], q: float) -> float:
    s = sorted(xs)
    if not s:
        return float("nan")
    i = (len(s) - 1) * q
    lo, hi = int(math.floor(i)), int(math.ceil(i))
    return s[lo] + (s[hi] - s[lo]) * (i - lo)


def won_of(row: dict) -> bool:
    """A game's kill: it won and was not capped (a capped game never counts as won)."""
    return bool(row.get("won")) and not row.get("capped")


HOPELESS_P = 0.95  # the search's own estimate of death at the fight's first decision from which it calls the position lost


def hopeless(row: dict) -> bool:
    """The search itself saw the fight as lost from the start: at least HOPELESS_P of its simulations at the first decision died."""
    return float(row.get("root_p_death") or 0.0) >= HOPELESS_P


def index_rows(rows: list[dict]) -> dict[str, dict[tuple, dict]]:
    """arm -> {(position, draw): row}; a later row of the same game replaces an earlier one (a rerun)."""
    out: dict[str, dict[tuple, dict]] = defaultdict(dict)
    for r in rows:
        if r.get("arm") is not None and r.get("pos") is not None:
            out[str(r["arm"])][(str(r["pos"]), int(r.get("j", 0)))] = r
    return out


def arm_summary(rows: list[dict]) -> dict:
    """One budget's line: games, kills, the interval, mean HP left and turns, capped, wall seconds and simulations."""
    n = len(rows)
    k = sum(1 for r in rows if won_of(r))
    lo, hi = wilson(k, n)
    wall = [float(r.get("seconds") or 0.0) for r in rows]
    won_hp = [float(r.get("end_hp") or 0) for r in rows if won_of(r)]
    return {"n": n, "kills": k, "rate": k / n if n else None, "lo": lo, "hi": hi, "capped": sum(1 for r in rows if r.get("capped")),
            "wall_mean": statistics.fmean(wall) if wall else None, "wall_median": statistics.median(wall) if wall else None,
            "wall_max": max(wall) if wall else None, "wall_sum": sum(wall),
            "sims_mean": statistics.fmean([float(r.get("sims") or 0) for r in rows]) if rows else None,
            "hp_left_when_won": statistics.fmean(won_hp) if won_hp else None,
            "hopeless": sum(1 for r in rows if hopeless(r)), "hopeless_kills": sum(1 for r in rows if hopeless(r) and won_of(r)),
            "turns_mean": statistics.fmean([float(r.get("turns") or 0) for r in rows]) if rows else None,
            "decisions_mean": statistics.fmean([float(r.get("decisions") or 0) for r in rows]) if rows else None}


def paired(idx: dict, a: str, b: str, keep=None, boot: int = BOOT, seed: int = BOOT_SEED) -> dict:
    """Budget `b` against budget `a` on the games both have (and `keep(row)` allows): kills of each, the discordant counts
    (only b killed: `up`; only a killed: `down`), the pairs that played the very same moves (`same_play`), the mean difference b - a with a 95% bootstrap interval over positions (all draws
    of a position move together), and the sign test's p."""
    pairs = []
    same = 0  # the pairs in which both budgets played the very same moves
    for key, ra in idx.get(a, {}).items():
        rb = idx.get(b, {}).get(key)
        if rb is None or (keep is not None and not keep(ra)):
            continue
        pairs.append((key[0], won_of(ra), won_of(rb)))
        same += bool(ra.get("moves_sha")) and ra.get("moves_sha") == rb.get("moves_sha")
    n = len(pairs)
    up = sum(1 for _, x, y in pairs if y and not x)
    down = sum(1 for _, x, y in pairs if x and not y)
    out = {"a": a, "b": b, "n": n, "kills_a": sum(1 for _, x, _y in pairs if x), "kills_b": sum(1 for _, _x, y in pairs if y), "up": up,
           "down": down, "same_play": same, "diff": (up - down) / n if n else None, "p_sign": sign_test_p(up, down), "lo": None, "hi": None}
    if n:
        by_pos: dict[str, list[int]] = defaultdict(list)
        for pos, x, y in pairs:
            by_pos[pos].append(int(y) - int(x))
        groups = list(by_pos.values())
        rng = random.Random(seed)
        means = []
        for _ in range(boot):
            tot = cnt = 0
            for _g in range(len(groups)):
                g = groups[rng.randrange(len(groups))]
                tot += sum(g)
                cnt += len(g)
            means.append(tot / cnt)
        out["lo"], out["hi"] = percentile(means, 0.025), percentile(means, 0.975)
    return out


def estimate(mean_wall: dict[int, float], counts: dict[int, int], procs: int) -> dict:
    """The full run's size from measured mean wall seconds a game per budget and the games per budget: core-hours and the wall
    hours on `procs` processes (the games are independent, the longest first, so the wall is the core-hours over the processes plus
    one longest game's tail, which the caller adds from the measured maximum)."""
    core_s = sum(float(mean_wall.get(m, 0.0)) * int(n) for m, n in counts.items())
    return {"core_hours": core_s / 3600.0, "wall_hours": core_s / 3600.0 / max(1, procs)}


def size_plan(mean_wall: dict[int, float], n_positions: int, core_hours: float, procs: int, wall_hours: float,
              floor_n16: int = 20, ceil_n: int | None = None) -> dict:
    """How many positions to play at each budget so the full run fits `core_hours` and `wall_hours` on `procs` processes.
    The positions are nested (the first N16 of the order are in the first N4, which are in the first N1), so every paired
    difference is on a common set. All budgets get the same N when that fits; else the dearest budget is cut first (never under
    `floor_n16`), then the next."""
    budget_s = min(core_hours * 3600.0, wall_hours * 3600.0 * procs)
    w = {m: float(mean_wall.get(m) or 0.0) for m in MULTS}
    ceil_n = min(ceil_n or n_positions, n_positions)
    n = {m: ceil_n for m in MULTS}
    cost = lambda cn: sum(w[m] * cn[m] for m in MULTS)  # noqa: E731
    if cost(n) <= budget_s:
        return {"n": n, **estimate(w, n, procs)}
    for m in sorted(MULTS, reverse=True):  # cut the dearest budget first, down to its floor
        lo = min(floor_n16, ceil_n)
        while n[m] > lo and cost(n) > budget_s:
            n[m] -= 1
    if cost(n) > budget_s:  # still too dear: cut every budget in proportion
        scale = budget_s / cost(n)
        n = {m: max(1, int(n[m] * scale)) for m in MULTS}
    # nested: a dearer budget never plays more positions than a cheaper one
    for hi, lo_m in ((4, 1), (16, 4)):
        n[hi] = min(n[hi], n[lo_m])
    return {"n": n, **estimate(w, n, procs)}


# ---------------------------------------------------------------------------------------------------------- files


def read_jsonl(p: Path) -> list[dict]:
    out = []
    if not Path(p).is_file():
        return out
    with open(p, encoding="utf-8") as fh:
        for x in fh:
            if x.strip():
                try:
                    out.append(json.loads(x))
                except ValueError:
                    pass  # a line cut by a stop: its game is played again
    return out


def iter_jsonl(p: Path):
    if not Path(p).is_file():
        return
    with open(p, encoding="utf-8") as fh:
        for x in fh:
            if x.strip():
                try:
                    yield json.loads(x)
                except ValueError:
                    pass


def say(out: Path, msg: str) -> None:
    import datetime as dt  # noqa: PLC0415

    with open(Path(out) / FILES["log"], "a", encoding="utf-8") as fh:
        fh.write(f"{dt.datetime.now(dt.timezone.utc).strftime('%Y-%m-%dT%H:%M:%SZ')} {msg}\n")


def write_json(p: Path, obj) -> None:
    tmp = Path(str(p) + ".tmp")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8")
    os.replace(tmp, p)


# ---------------------------------------------------------------------------------------------------------- the positions


def store_name(store: Path) -> str:
    return Path(store).name


def candidates(stores: list[Path], salt: int = SEED):
    """(ordinal -> list of (order key, store, generation, seed)) of the runs that met an Act 3 boss (ordinal 1) and those that met a
    second one (ordinal 2), from the rows of the stores' runs.jsonl; a seed once per store; capped rows left out."""
    by_ord: dict[int, list] = {1: [], 2: []}
    for store in stores:
        gen = store_name(store)
        seen: set[str] = set()
        for row in iter_jsonl(Path(store) / "runs.jsonl"):
            seed = row.get("seed")
            if not seed or row.get("capped") or seed in seen:
                continue
            a3 = [b for b in row.get("bosses") or [] if int(b.get("act", -1)) == ACT3_INDEX]
            if not a3:
                continue
            seen.add(seed)
            for o in (1, 2):
                if len(a3) >= o:
                    by_ord[o].append((order_key(salt, o, gen, seed), str(store), gen, str(seed)))
    for o in by_ord:
        by_ord[o].sort()
    return by_ord


def record_path(store: str, seed: str) -> Path:
    return Path(store) / "decisions.d" / f"{seed}.record.json"


def native_handle():
    import gen0  # noqa: PLC0415

    gen0._setup()  # noqa: SLF001 -- the library path and the stale-library allowance
    from sts2_gym import native  # noqa: PLC0415

    return native


def build_doc(spec: dict, native=None) -> tuple[dict | None, str]:
    """The fight's opening as an `sts2-emu-snapshot/1` document (the copy's arena entry, then its exported state), the run's relic
    counters written in. (document, "") or (None, reason)."""
    import ctypes  # noqa: PLC0415

    native = native or native_handle()
    h = None
    try:
        h = native.create(game_seed_of(spec["id"], "a3build"))
        obs = (ctypes.c_int * native.OBS_SIZE)()
        a = reset_args(spec)
        native.reset_arena(h, a.pop("deck_ids"), obs, **a)
        doc = native.get_state_json(h)
    except Exception as error:  # noqa: BLE001 -- the copy refused the position: counted by its message
        return None, f"copy refused: {type(error).__name__}: {str(error)[:80]}"
    finally:
        if h is not None:
            native.destroy(h)
    if int(doc.get("encounter_id", -1)) != int(spec["encounter_id"]) or not doc.get("enemies"):
        return None, "the opening has another boss or none"
    spec["counters_written"] = apply_counters(doc, spec["counters"])
    return doc, ""


def build_positions(out: Path, stores: list[Path], n_first: int, n_second: int, salt: int = SEED, native=None) -> dict:
    """Pick the positions (a deterministic hash order, the first `n` of each ordinal whose record and opening build) and write
    positions.jsonl (one position a line with its opening document) and meta.json."""
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    cands = candidates(stores, salt)
    meta = {"rule": RULE, "salt": salt, "stores": [str(s) for s in stores], "asked": {"first": n_first, "second": n_second},
            "candidates": {"first": len(cands[1]), "second": len(cands[2])}, "skipped": defaultdict(int)}
    chosen: list[dict] = []
    seen_ids: set[str] = set()
    for ordinal, want in ((1, n_first), (2, n_second)):
        got = 0
        for _key, store, gen, seed in cands[ordinal]:
            if got >= want:
                break
            p = record_path(store, seed)
            try:
                record = json.loads(p.read_text(encoding="utf-8"))
            except (OSError, ValueError):
                meta["skipped"]["record unreadable"] += 1
                continue
            spec = position_spec(record, ordinal, gen, store)
            if spec is None:
                meta["skipped"]["fight not in the record"] += 1
                continue
            if spec["id"] in seen_ids:
                continue
            doc, why = build_doc(spec, native)
            if doc is None:
                meta["skipped"][why] += 1
                continue
            spec["doc"] = doc
            chosen.append(spec)
            seen_ids.add(spec["id"])
            got += 1
    for rank, s in enumerate(sorted(chosen, key=lambda s: order_key(salt, 0, "all", s["id"]))):
        s["order"] = rank  # one hash order over every chosen position: a prefix is a random subset with the sample's own mix
    tmp = out / (FILES["positions"] + ".tmp")
    with open(tmp, "w", encoding="utf-8") as fh:
        for s in sorted(chosen, key=lambda s: s["order"]):
            fh.write(json.dumps(s, separators=(",", ":")) + "\n")
    os.replace(tmp, out / FILES["positions"])
    meta["skipped"] = dict(meta["skipped"])
    meta["positions"] = len(chosen)
    meta["by_ordinal"] = {str(o): sum(1 for s in chosen if s["ordinal"] == o) for o in (1, 2)}
    meta["by_boss"] = {b: sum(1 for s in chosen if s["boss"] == b) for b in sorted({s["boss"] for s in chosen})}
    meta["by_generation"] = {str(g): sum(1 for s in chosen if str(s["gen"]) == str(g)) for g in sorted({str(s["gen"]) for s in chosen})}
    meta["run_won"] = sum(1 for s in chosen if s["run_won"])
    meta["with_counters_written"] = sum(1 for s in chosen if s.get("counters_written"))
    write_json(out / FILES["meta"], meta)
    return meta


def load_positions(out: Path) -> dict[str, dict]:
    return {s["id"]: s for s in read_jsonl(Path(out) / FILES["positions"])}


# ---------------------------------------------------------------------------------------------------------- the games (child side)


def _init_worker(engine: str, agent_dir: str, out: str, lib: str, cfgs: dict) -> None:
    """A worker's one-off: BelowNormal, the engine and its library, the champion, the positions (documents) and the arms' configs."""
    sys.path.insert(0, str(ROOT))
    from tools.solver import coverage  # noqa: PLC0415

    coverage._below_normal()  # noqa: SLF001 -- Windows priority class
    if lib:
        os.environ["STS2_LIB_PATH"] = lib
    import gen0  # noqa: PLC0415

    gen0.select_engine(engine)
    gen0._setup()  # noqa: SLF001
    import fight_gym as fg  # noqa: PLC0415
    import gen_loop  # noqa: PLC0415

    fg._W["agent"] = gen_loop.load_agent(agent_dir)  # noqa: SLF001
    fg._W["act3_positions"] = load_positions(Path(out))  # noqa: SLF001
    fg._W["act3_cfgs"] = cfgs  # noqa: SLF001
    fg._W["act3_engine"] = engine  # noqa: SLF001


def game_task(task: tuple) -> dict:
    """One game: position `pid` at budget `arm` (`<m>x`), draw j, from the opening document."""
    import fight_gym as fg  # noqa: PLC0415

    arm, pid, j = task[:3]
    W = fg._W  # noqa: SLF001
    start = W["act3_positions"][pid]
    cfg = W["act3_cfgs"][arm]
    mult = int(arm[:-1])
    extra = W["agent"].search_extra()
    world = fg._fresh_world(start["doc"], pid, j, NS)  # noqa: SLF001
    try:
        row = fg._play(world, start, j, cfg, extra, NS, deadline=DEADLINE_S.get(mult, DEADLINE_S[16]))  # noqa: SLF001
    finally:
        world.close()
    mem = fg.private_bytes(os.getpid())
    return {"_task": [arm, pid, j], "arm": arm, "mult": mult, "pos": pid, "j": j, "ordinal": start["ordinal"], "boss": start["boss"],
            "gen": start["gen"], "run_won": start["run_won"], "start_hp": start["hp"], "max_hp_start": start["max_hp"],
            "config": cfg.name, "engine": W.get("act3_engine"), "rule": RULE, "mem_mb": round(mem / 2**20) if mem else None, **row}


def hard_deadline(task: tuple) -> float:
    return DEADLINE_S.get(int(task[0][:-1]), DEADLINE_S[16]) * 1.5 + 60.0


def capped_row(task: tuple, reason: str, seconds: float, prog: dict) -> dict:
    """The row of a game its worker could not finish: capped, not won, with how far it got."""
    arm, pid, j = task[:3]
    return {"_task": [arm, pid, j], "arm": arm, "mult": int(arm[:-1]), "pos": pid, "j": j, "capped": True, "cap_reason": reason,
            "seconds": round(seconds, 2), "won": False, "end_hp": 0, "max_hp": prog.get("max_hp"), "turns": prog.get("turns", 1),
            "decisions": prog.get("k", 0), "sims": prog.get("sims", 0), "rule": RULE}


# ---------------------------------------------------------------------------------------------------------- playing


def champion(agent_dir: str | None) -> tuple[Path, object]:
    """(the champion's folder, its Agent): `agent_dir`, else what referee/agent/current.json names."""
    import gen_loop  # noqa: PLC0415

    if agent_dir:
        d = Path(agent_dir)
    else:
        d = Path(json.loads(Path(DEFAULT_AGENT_CURRENT).read_text(encoding="utf-8"))["dir"])
    return d, gen_loop.load_agent(d)


def tasks_for(positions: dict[str, dict], counts: dict[int, int], draws: int, done: set, order: str = "dearest") -> list[tuple]:
    """The games still to play. Positions are nested by their `order` (a hash order over all of them): a budget with count n plays
    the first n. `order`: "dearest" starts the long games first (the full run: the tail is the short ones), "cheapest" plays every
    position at the cheap budgets before the dear ones (a smoke: a stop leaves the dear budget with fewer positions, never the cheap
    one), "position" finishes one position at every budget before the next (a stop leaves whole positions)."""
    ranked = sorted(positions.values(), key=lambda s: (s["order"], s["id"]))
    tasks = []
    for mult in counts:
        for rank, s in enumerate(ranked[: counts[mult]]):
            for j in range(draws):
                if (arm_label(mult), s["id"], j) not in done:
                    tasks.append((rank, mult, j, (arm_label(mult), s["id"], j)))
    if order == "dearest":
        tasks.sort(key=lambda t: (-t[1], t[0], t[2]))
    elif order == "cheapest":
        tasks.sort(key=lambda t: (t[1], t[0], t[2]))
    elif order == "position":
        tasks.sort(key=lambda t: (t[0], -t[1], t[2]))
    else:
        raise ValueError(f"order {order!r}: dearest, cheapest or position")
    return [t[3] for t in tasks]


def done_keys(rows: list[dict]) -> set:
    return {(str(r["_task"][0]), str(r["_task"][1]), int(r["_task"][2])) for r in rows if r.get("_task")}


def parse_counts(text: str | None, mults: list[int], n_pos: int) -> dict[int, int]:
    """`--n-per-mult 1:60,4:60,16:20` as {mult: positions}; without it every budget plays every position."""
    if not text:
        return {m: n_pos for m in mults}
    out = {}
    for part in text.split(","):
        m, n = part.split(":")
        out[int(m)] = min(int(n), n_pos)
    return {m: out.get(m, 0) for m in mults}


def cmd_play(a: argparse.Namespace) -> int:
    import fight_exam as fe  # noqa: PLC0415
    import fight_gym as fg  # noqa: PLC0415
    import gym_loop as gl  # noqa: PLC0415

    out = Path(a.out)
    positions = load_positions(out)
    if not positions:
        print(f"no positions in {out}: run `act3_replay.py positions` first")
        return 2
    agent_dir, agent = champion(a.agent_dir)
    stakes, hold = agent.stakes, agent.hold
    mults = sorted(int(m) for m in a.mults)
    cfgs = {arm_label(m): arm_config(stakes, hold, m) for m in mults}
    counts = parse_counts(a.n_per_mult, mults, a.limit_pos or len(positions))
    games = out / FILES["games"]
    done = done_keys(read_jsonl(games))
    tasks = tasks_for(positions, counts, a.draws, done, a.order)
    if not tasks:
        print("all games are played")
        return 0
    for d in DEADLINE_S:
        if a.deadline_scale != 1.0:
            DEADLINE_S[d] *= a.deadline_scale
    fg.MAX_PROCS = max(fg.MAX_PROCS, a.procs)
    ctx = {"engine": a.engine, "agent": str(agent_dir), "configs": {k: v.name for k, v in cfgs.items()}, "stakes": stakes, "procs": a.procs}
    write_json(out / "play-context.json", {**ctx, "started": gl.stamp()})
    say(out, f"playing {len(tasks)} games ({counts} positions per budget, {a.draws} draws) on {a.procs} processes under {ctx['configs']}")
    fe.ALLOW_QUIET.append(TAG)
    started = time.time()
    until = a.until or None
    if a.wall_hours:
        import datetime as dt  # noqa: PLC0415

        cut = (dt.datetime.now(dt.timezone.utc) + dt.timedelta(hours=a.wall_hours)).strftime("%Y-%m-%dT%H:%M:%SZ")
        until = min(filter(None, [until, cut]))

    def stop() -> str | None:
        return fe.blocked_reason(starting=False, until=until, stop_file=out / "STOP")

    why = fe.blocked_reason(starting=True, until=until, stop_file=out / "STOP")
    if why:
        print(f"not started: {why}")
        say(out, f"not started: {why}")
        return 3
    try:
        fe.prepare_process()
    except fe.Blocked as why2:
        print(f"not started: {why2}")
        return 3
    stopped = fg.run_tasks(tasks, game_task, games, a.procs, init=_init_worker,
                           init_args=(a.engine, str(agent_dir), str(out), a.lib or "", cfgs), hard_of=hard_deadline,
                           mem_limit=a.mem_gb * 2**30, capped_of=capped_row, stop=stop)
    left = len(tasks_for(positions, counts, a.draws, done_keys(read_jsonl(games)), a.order))
    say(out, f"{'stopped (' + stopped + ')' if stopped else 'finished'} after {time.time() - started:.0f} s; {left} games left")
    print(f"stopped ({stopped}); {left} games left" if stopped else f"played; {left} left")
    return 3 if stopped or left else 0


# ---------------------------------------------------------------------------------------------------------- the report


def report_data(out: Path) -> dict:
    out = Path(out)
    pos = load_positions(out)
    rows = read_jsonl(out / FILES["games"])
    idx = index_rows(rows)
    arms = sorted(idx, key=lambda s: int(s[:-1]))
    data = {"positions": len(pos), "arms": {}, "paired": [], "splits": {}}
    for arm in arms:
        data["arms"][arm] = arm_summary(list(idx[arm].values()))
    for a, b in ((x, y) for i, x in enumerate(arms) for y in arms[i + 1:]):
        data["paired"].append(paired(idx, a, b))
    splits = {"boss": lambda r: str(r.get("boss")), "ordinal": lambda r: f"boss {r.get('ordinal')}",
              "run_result": lambda r: "the run beat it" if r.get("run_won") else "the run lost to it"}
    for name, fn in splits.items():
        cells = sorted({fn(r) for arm in arms for r in idx[arm].values()})
        data["splits"][name] = {}
        for cell in cells:
            data["splits"][name][cell] = {"arms": {arm: arm_summary([r for r in idx[arm].values() if fn(r) == cell]) for arm in arms},
                                          "paired": [paired(idx, a, b, keep=lambda r, c=cell, f=fn: f(r) == c, boot=500)
                                                     for a, b in ((x, y) for i, x in enumerate(arms) for y in arms[i + 1:])]}
    return data


def pct(x) -> str:
    return "-" if x is None else f"{100 * x:.1f}"


def report_text(d: dict) -> str:
    L = [f"Act 3 boss replay: {d['positions']} positions", ""]
    L.append("Kills per budget (in 100; 95% interval)   games  kills  capped   wall s mean / median / max   sims mean   HP left when won")
    for arm, s in d["arms"].items():
        L.append(f"  {arm:>4}  {pct(s['rate']):>5}  [{pct(s['lo'])}, {pct(s['hi'])}]   n={s['n']:<4} k={s['kills']:<4} cap={s['capped']:<3}"
                 f"  {s['wall_mean']:.0f} / {s['wall_median']:.0f} / {s['wall_max']:.0f}   {s['sims_mean']:.0f}   "
                 f"{'-' if s['hp_left_when_won'] is None else format(s['hp_left_when_won'], '.1f')}")
    L += ["", "Positions the search itself calls lost at the first decision (95 in 100 of its simulations die), and the kills among them"]
    for arm, s in d["arms"].items():
        L.append(f"  {arm:>4}  {s['hopeless']} of {s['n']} games; kills among them {s['hopeless_kills']}; kills among the rest {s['kills'] - s['hopeless_kills']} of {s['n'] - s['hopeless']}")
    L += ["", "Paired differences on the games both budgets have (in 100; 95% bootstrap interval over positions; sign test p)"]
    for p in d["paired"]:
        L.append(f"  {p['b']} minus {p['a']}: {pct(p['diff'])} [{pct(p['lo'])}, {pct(p['hi'])}]  n={p['n']}  "
                 f"kills {p['kills_a']} -> {p['kills_b']} (only b {p['up']}, only a {p['down']})  p={p['p_sign']:.3f}  same moves in {p['same_play']} of {p['n']}")
    for name, cells in d["splits"].items():
        L += ["", f"By {name}"]
        for cell, c in cells.items():
            line = "  ".join(f"{arm} {s['kills']}/{s['n']} ({pct(s['rate'])})" for arm, s in c["arms"].items())
            diffs = "  ".join(f"{p['b']}-{p['a']} {pct(p['diff'])}" for p in c["paired"] if p["n"])
            L.append(f"  {cell}: {line}   |  {diffs}")
    return "\n".join(L)


def cmd_report(a: argparse.Namespace) -> int:
    d = report_data(Path(a.out))
    print(json.dumps(d, indent=1) if a.json else report_text(d))
    return 0


def cmd_size(a: argparse.Namespace) -> int:
    d = report_data(Path(a.out))
    mean_wall = {int(arm[:-1]): s["wall_mean"] for arm, s in d["arms"].items() if s["wall_mean"]}
    missing = [m for m in MULTS if m not in mean_wall]
    if missing:
        print(f"no measured wall for budgets {missing}")
        return 2
    plan = size_plan(mean_wall, a.positions, a.core_hours, a.procs, a.wall_hours, ceil_n=a.positions)
    print(json.dumps({"mean_wall_s": mean_wall, **plan}, indent=1))
    return 0


def cmd_positions(a: argparse.Namespace) -> int:
    import gen0  # noqa: PLC0415

    if a.lib:
        os.environ["STS2_LIB_PATH"] = a.lib
    gen0.select_engine(a.engine)
    meta = build_positions(Path(a.out), [Path(s) for s in a.stores], a.n_first, a.n_second, a.salt)
    print(json.dumps({k: meta[k] for k in ("positions", "by_ordinal", "by_boss", "by_generation", "run_won", "with_counters_written",
                                            "skipped", "candidates")}, indent=1))
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("positions", help="pick the Act 3 boss positions from the stores and build their openings")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--stores", nargs="+", default=list(DEFAULT_STORES))
    p.add_argument("--n-first", type=int, default=30)
    p.add_argument("--n-second", type=int, default=8)
    p.add_argument("--salt", type=int, default=SEED)
    p.add_argument("--engine", choices=("old", "core2"), default="old")
    p.add_argument("--lib", default="", help="the folder of Sts2Emulator.dll (default: this tree's out/)")
    p.set_defaults(fn=cmd_positions)
    p = sub.add_parser("play", help="play every position at every budget (resumable)")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--agent-dir", default="", help="the champion's folder (default: referee/agent/current.json)")
    p.add_argument("--mults", nargs="+", type=int, default=list(MULTS))
    p.add_argument("--n-per-mult", default="", help="positions per budget, e.g. 1:60,4:60,16:20 (nested by the positions' order)")
    p.add_argument("--limit-pos", type=int, default=0)
    p.add_argument("--draws", type=int, default=1)
    p.add_argument("--order", choices=("dearest", "cheapest", "position"), default="dearest")
    p.add_argument("--procs", type=int, default=4)
    p.add_argument("--engine", choices=("old", "core2"), default="old")
    p.add_argument("--lib", default="")
    p.add_argument("--mem-gb", type=float, default=2.5, help="a worker over this private memory is killed and its game capped")
    p.add_argument("--until", default="", help="a UTC stamp YYYY-MM-DDTHH:MM:SSZ after which no game starts")
    p.add_argument("--wall-hours", type=float, default=0.0)
    p.add_argument("--deadline-scale", type=float, default=1.0)
    p.set_defaults(fn=cmd_play)
    p = sub.add_parser("report", help="the table of a run")
    p.add_argument("--out", default=DEFAULT_OUT)
    p.add_argument("--json", action="store_true")
    p.set_defaults(fn=cmd_report)
    p = sub.add_parser("size", help="the full run's size from a measured sample")
    p.add_argument("--out", required=True)
    p.add_argument("--positions", type=int, default=300)
    p.add_argument("--core-hours", type=float, default=90.0)
    p.add_argument("--procs", type=int, default=12)
    p.add_argument("--wall-hours", type=float, default=8.0)
    p.set_defaults(fn=cmd_size)
    a = ap.parse_args(argv)
    return a.fn(a)


if __name__ == "__main__":
    sys.exit(main())
