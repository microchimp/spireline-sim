#!/usr/bin/env python3
"""The kill check against its config's base on the fight board (2026-11-04): a3-v1-k and a3-v1, one network, the same games.

    python rl/kill_check_board.py --out DIR [--player v2] [--starts 200] [--seeds 2] [--procs 6] [--k 16] [--budget-min 30]
    python rl/kill_check_board.py --out DIR --summarize          (the figures from the games already in DIR/games.jsonl)

Plays the board's Act-1 boss starts (`gym_loop.board_starts`, the gate's 200) under each arm for draw seeds 0 .. `--seeds` - 1, the
arms' games interleaved in one lockstep run (`gym_lockstep.run_lockstep`, the CPU prior of PLAYER, BelowNormal) so both meet the same
machine. A game's seeds derive from (start, j, decision index) only, so the arms are paired: until the kill check fires the two
arms play the same moves. Each game's row is `fight_gym.play_steps`' (a3-v1-k rows add the check's counts); the run is resumable
(rows are appended, games with a row are not played again).

`--shadow` (the default) sets SPIRELINE_LETHAL_SHADOW: wherever the check fires it also runs the plain search from the same seed
and records the move that search chose (`lethal_differs` lists the decisions where the two moves differ). It adds the plain
search's time to the game's own clock, not to `search_s` (the search seconds the figures below are made of).

Heavy: take a QUIET.lock line (tag `kill-check`) before running it. It stops by itself at the budget, when the night's marker
appears, when a STOP file is put in DIR, or when another lane's QUIET.lock line stands (the fight loop's does not count).
Lab code: nothing here changes the engine, the search or a config.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
for _p in (ROOT / "rl", ROOT / "src", ROOT):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import fight_gym as fg  # noqa: E402

ARMS = ("a3-v1", "a3-v1-k")  # the base first, the kill-checking config second
TAG = "kill-check"


def read_games(path: Path) -> list[dict]:
    """The games in `path`, the last row of a (config, start, j) when it was played twice."""
    last: dict[tuple, dict] = {}
    for r in fg.read_jsonl(path):
        last[(r.get("config"), r.get("start"), int(r.get("j", 0)))] = r
    return list(last.values())


def build_tasks(starts: list[dict], seeds: int, player: str, starts_file: str, ns: str, done: set[tuple]) -> list[tuple]:
    """Every (arm, start, j) without a row, j-major, the two arms of a game next to each other."""
    info = {"starts": starts_file, "incoming": "", "player": player, "record": False, "ns": ns}
    return [("a3-v1", s["id"], j, s["cat"], {**info, "config": cfg, "label": cfg})
            for j in range(seeds) for s in starts for cfg in ARMS if (cfg, s["id"], j) not in done]


def exact_two_sided(b: int, c: int) -> float:
    """Exact two-sided McNemar test on b and c discordant pairs (1.0 when there are none)."""
    n = b + c
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, i) for i in range(min(b, c) + 1)) / 2**n
    return min(1.0, 2 * tail)


def won(r: dict) -> bool:
    return bool(r.get("won")) and not r.get("capped")


def arm_figures(rows: list[dict]) -> dict:
    n = len(rows)
    decisions = sum(int(r.get("decisions") or 0) for r in rows)
    search_s = sum(float(r.get("search_s") or 0.0) for r in rows)
    sims = sum(int(r.get("sims") or 0) for r in rows)
    wins = sum(won(r) for r in rows)
    out = {"games": n, "wins": wins, "win_rate": wins / n if n else None,
           "mean_L0": sum(float(r.get("L0") or 0.0) for r in rows) / n if n else None,
           "mean_end_hp_if_won": (sum(float(r.get("end_hp") or 0) for r in rows if won(r)) / wins) if wins else None,
           "capped": sum(bool(r.get("capped")) for r in rows), "decisions": decisions,
           "search_s": round(search_s, 2), "search_ms_per_decision": round(1000 * search_s / decisions, 3) if decisions else None,
           "sims": sims, "sims_per_decision": round(sims / decisions, 2) if decisions else None,
           "decisions_per_game": round(decisions / n, 2) if n else None,
           "game_seconds_mean": round(sum(float(r.get("seconds") or 0.0) for r in rows) / n, 3) if n else None}
    if any("lethal_checked" in r for r in rows):
        checked = sum(int(r.get("lethal_checked") or 0) for r in rows)
        fired = sum(int(r.get("lethal_fired") or 0) for r in rows)
        lethal_s = sum(float(r.get("lethal_s") or 0.0) for r in rows)
        out.update(checked=checked, fired=fired, fire_rate=fired / checked if checked else None,
                   fire_rate_of_decisions=fired / decisions if decisions else None,
                   games_with_a_fire=sum(1 for r in rows if int(r.get("lethal_fired") or 0) > 0),
                   check_ms_per_look=round(1000 * lethal_s / checked, 3) if checked else None, check_s=round(lethal_s, 2))
    return out


def summarize(rows: list[dict]) -> dict:
    """The figures of the head-to-head: each arm's table, the paired comparison, the check's fires and the decisions where its
    move differed from the search's. `rows` are the games of both arms."""
    by: dict[str, dict[tuple, dict]] = {a: {} for a in ARMS}
    for r in rows:
        if r.get("config") in by:
            by[r["config"]][(r["start"], int(r["j"]))] = r
    base, kill = by[ARMS[0]], by[ARMS[1]]
    keys = sorted(set(base) & set(kill))
    wb = [won(base[x]) for x in keys]
    wk = [won(kill[x]) for x in keys]
    only_k = sum(1 for x, y in zip(wk, wb) if x and not y)
    only_b = sum(1 for x, y in zip(wk, wb) if y and not x)
    d = [float(x) - float(y) for x, y in zip(wk, wb)]
    mean, se, _ = fg.mean_se(d) if d else (None, None, 0)
    identical = sum(1 for x in keys if base[x].get("moves_sha") == kill[x].get("moves_sha"))
    differs, wrong = [], []
    for x in keys:
        for dd in kill[x].get("lethal_differs") or []:
            differs.append({"start": x[0], "j": x[1], **dd})
        for dd in kill[x].get("lethal_wrong") or []:
            wrong.append({"start": x[0], "j": x[1], **dd})
    fired_games = [x for x in keys if int(kill[x].get("lethal_fired") or 0) > 0]
    return {
        "arms": {a: arm_figures([by[a][x] for x in keys]) for a in ARMS},
        "all_games": {a: arm_figures(list(by[a].values())) for a in ARMS},
        "paired": {"pairs": len(keys), "wins_base": sum(wb), "wins_kill": sum(wk), "kill_only": only_k, "base_only": only_b,
                   "win_diff": mean, "win_se": se, "p_two_sided": exact_two_sided(only_k, only_b),
                   "games_played_alike": identical, "games_with_a_fire": len(fired_games),
                   "L0_diff": (sum(float(kill[x].get("L0") or 0.0) - float(base[x].get("L0") or 0.0) for x in keys) / len(keys)) if keys else None},
        "differs": {"count": len(differs), "decisions": differs},
        "wrong_kills": {"count": len(wrong), "decisions": wrong},  # fired kills that did not end the fight on the true position (--shadow)
    }


def report_lines(s: dict) -> list[str]:
    a, b, p = s["arms"][ARMS[0]], s["arms"][ARMS[1]], s["paired"]
    f = (lambda v, nd=2: "n/a" if v is None else f"{v:.{nd}f}")  # noqa: E731
    return [
        f"pairs {p['pairs']}: wins {ARMS[0]} {p['wins_base']}, {ARMS[1]} {p['wins_kill']} (kill-only {p['kill_only']}, base-only {p['base_only']}, "
        f"diff {100 * (p['win_diff'] or 0):+.2f} points, SE {100 * (p['win_se'] or 0):.2f}, p {f(p['p_two_sided'], 3)}); L0 diff {f(p['L0_diff'], 4)}",
        f"the check looked {b.get('checked') or 0} times and fired {b.get('fired') or 0} ({f(100 * (b.get('fire_rate') or 0), 1)}% of looks, "
        f"{f(100 * (b.get('fire_rate_of_decisions') or 0), 1)}% of decisions) in {b.get('games_with_a_fire') or 0} of {b['games']} games; "
        f"a look costs {f(b.get('check_ms_per_look'), 2)} ms",
        f"search seconds per decision: {ARMS[0]} {f(a['search_ms_per_decision'], 2)} ms, {ARMS[1]} {f(b['search_ms_per_decision'], 2)} ms "
        f"(speed gain {f(100 * (1 - b['search_ms_per_decision'] / a['search_ms_per_decision']), 1) if a['search_ms_per_decision'] and b['search_ms_per_decision'] else 'n/a'}%, positive is faster); "
        f"sims per decision {f(a['sims_per_decision'])} against {f(b['sims_per_decision'])}",
        f"decisions where the kill check's move differed from the search's: {s['differs']['count']}; fired kills that did not hold on the true "
        f"position: {s['wrong_kills']['count']}; games played move for move alike {p['games_played_alike']} of {p['pairs']}",
    ]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--out", required=True, help="the results folder: games.jsonl, summary.json")
    ap.add_argument("--player", default="v2", help="the network: a folder under --players")
    ap.add_argument("--players", default=str(fg.REF / "gym-loop" / "players"))
    ap.add_argument("--starts", type=int, default=200)
    ap.add_argument("--seeds", type=int, default=2, help="draw seeds j = 0 .. N-1, paired across the arms")
    ap.add_argument("--procs", type=int, default=6)
    ap.add_argument("--k", type=int, default=16, help="fights a lockstep process holds")
    ap.add_argument("--budget-min", type=float, default=30.0)
    ap.add_argument("--quiet-tag", default=TAG, help="this job's own QUIET.lock tag: its line never stops it")
    ap.add_argument("--no-shadow", action="store_true")
    ap.add_argument("--summarize", action="store_true")
    a = ap.parse_args(argv)
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    target = out / "games.jsonl"
    if not a.summarize:
        import gym_loop as gl  # noqa: PLC0415
        import gym_lockstep  # noqa: PLC0415

        from tools.lab import winsys  # noqa: PLC0415

        winsys.set_below_normal()
        if not a.no_shadow:
            os.environ["SPIRELINE_LETHAL_SHADOW"] = "1"
        starts = gl.board_starts()[: a.starts]
        done = {(r.get("config"), r.get("start"), int(r.get("j", 0))) for r in fg.read_jsonl(target)}
        tasks = build_tasks(starts, a.seeds, a.player, str(gl.BOARD_DIR / "starts.jsonl"), gl.PROMOTION_RULE["ns"], done)
        print(f"{len(starts)} starts x {a.seeds} seeds x {len(ARMS)} arms: {len(done)} games on disk, {len(tasks)} to play", flush=True)
        t0 = time.time()

        def stop() -> str | None:
            if (out / "STOP").exists():
                return "stop file"
            if fg.ACTIVE.exists():
                return "nightly-active"
            if time.time() - t0 > a.budget_min * 60:
                return "budget"
            tag = gl.other_quiet(allow=(a.quiet_tag,))
            return f"quiet:{tag}" if tag else None

        why = None
        if tasks:
            init_args = (gl.ENGINE, a.players, a.player, str(out))
            opts = {"prior": "cpu", "server": None, "game_steps": "gym_loop:gen_game_steps", "worlds": "reload", "grade_wall_s": 0,
                    "paths": [str(ROOT / "rl")]}
            why, stats = gym_lockstep.run_lockstep(tasks, target, a.procs, min(a.k, gym_lockstep.K_MAX), init_args, opts, stop=stop,
                                                   log=lambda m: print(m, flush=True))
            print(f"run ended: {why or 'every game played'} after {time.time() - t0:.0f} s", flush=True)
    s = summarize(read_games(target))
    (out / "summary.json").write_text(json.dumps(s, indent=1), encoding="utf-8")
    for line in report_lines(s):
        print(line, flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
