"""Calibrate the `skipcal` chooser's skip bias (2026-11-06, docs/rl/STAKES_AND_HOLD.md).

The blended chooser (`imitation_chooser`) takes the argmax of V + lambda log pi_h on a card reward, and that argmax skips Act 1 card
rewards far less than humans do (5.4 in 100 on the first sanity run against the humans' 15.4: a card is the likelier option whenever
it beats the skip, and it nearly always does). The `skipcal<bias>` pick mode (`macro_recipe.pick_name`) adds `bias` floors to the skip
option's score. This tool finds the bias that makes the agent's Act 1 skip rate land within `TOLERANCE` (3 points) of the humans'
`HUMAN_ACT1_SKIP` (15.4 in 100) on a quick copy check of `--runs` (50) runs.

How, in rounds, each round a block of runs on the copy (one emulator per process, BelowNormal, the chooser's agent as built):

1. Play the block with the round's bias (round 0: 0, which is the argmax). Every run row of a `skipcal` agent carries, for each card
   screen with a skip option, the screen's skip gap (`decided`'s sixth entry): the bias at which the skip would have been taken there.
2. The round's skip rate is read off the rows' `cards` (skips over the card rewards decided on floors 1 to 17).
3. If it is within the tolerance the bias is the answer. Otherwise the next bias is the one that would have given the target rate on
   the gaps of the screens this round met (`bias_for`, an open-loop reading: the skip rate at bias b on a fixed set of screens is the
   share of gaps at or under b), and the loop goes round again: the screens move with the bias (more skips, other decks), so the
   answer is only an answer once a round at that very bias lands in the tolerance. Up to `--max-rounds` rounds.

The search is the plain `a3-v1` by default (the skip is a macro decision, and the stakes schedule and the hold only change the fights'
price; a 50-run check at x16 bosses would take hours); `--keep-search` plays the agent's own search. The result is a JSON line on
stdout and in `--out/calibration.json`: the bias, whether a round at it landed within the tolerance, and every round's bias, skip rate,
screens and the open-loop reading. Build the candidate with that bias: `gen0.py build ... --pick skipcal --skip-bias <bias>`.

    python rl/skip_calibrate.py --agent-dir <gen>/candidate-<seed candidate> --lib <out dir> --out <dir> [--runs 50] [--procs 2]

The seed candidate is any built candidate whose macro is `vrun+pi_h:...:pick=skipcal<any>`: only its human prior, lambda and run
judgement are used. Heavy (50 runs a round, 20 to 40 minutes on one process): hold a QUIET.lock line (`--quiet-tag`, default
`skipcal`); the always-on loop's `gym-loop` line does not block it.
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import os
import statistics
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
for _p in (str(ROOT / "rl"), str(ROOT / "src"), str(ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import macro_recipe  # noqa: E402

HUMAN_ACT1_SKIP = 0.154  # humans' share of Act 1 card rewards skipped (docs/rl/IMITATION_CHOOSER.md, the sanity run)
TOLERANCE = 0.03  # within 3 points of that
ACT1_LAST_FLOOR = 17  # the Act 1 boss is floor 17
MIN_SCREENS = 150  # a round with fewer Act 1 card screens than this is too thin to land a rate within the tolerance
BIAS_LIMIT = 12.0  # floors; a bias beyond this is a sign the gaps are wrong, not a calibration
QUIET_FILE = "D:/spireline/referee/QUIET.lock"


def skip_rate(rows: Sequence[dict]) -> tuple[float | None, int]:
    """(skips over the Act 1 card rewards the runs decided, how many such rewards) from the rows' `cards` ([[floor, "NAME" | "skip"]])."""
    skips = screens = 0
    for r in rows:
        for entry in r.get("cards") or []:
            floor, what = entry[0], entry[1]
            if isinstance(floor, int) and 1 <= floor <= ACT1_LAST_FLOOR:
                screens += 1
                skips += what == "skip"
    return (skips / screens if screens else None), screens


def act1_gaps(rows: Sequence[dict]) -> list[float]:
    """The skip gaps of the Act 1 card screens the runs met (`decided` entries [floor, "card", prior, v, guard, gap])."""
    out = []
    for r in rows:
        for d in r.get("decided") or []:
            if len(d) >= 6 and d[1] == "card" and 1 <= int(d[0]) <= ACT1_LAST_FLOOR:
                out.append(float(d[5]))
    return out


def rate_at(gaps: Sequence[float], bias: float) -> float:
    """The skip rate at `bias` on these screens: the share whose gap is at or under it."""
    if not gaps:
        raise ValueError("no screens")
    return sum(g <= bias for g in gaps) / len(gaps)


def bias_for(gaps: Sequence[float], target: float) -> float:
    """The bias at which `rate_at(gaps, bias)` is the share of `gaps` closest to `target`: between the gap that ends the first k
    screens (k = target times the screens, rounded) and the next one, rounded to the three decimals the recipe keeps."""
    if not gaps:
        raise ValueError("no screens to calibrate on")
    g = sorted(gaps)
    n = len(g)
    k = max(0, min(n, round(target * n)))
    if k == 0:
        b = g[0] - 0.05
    elif k == n:
        b = g[-1] + 0.05
    else:
        b = (g[k - 1] + g[k]) / 2
    return round(max(-BIAS_LIMIT, min(BIAS_LIMIT, b)), 3)


def calibrate(play: Callable[[float], list[dict]], *, target: float = HUMAN_ACT1_SKIP, tol: float = TOLERANCE, max_rounds: int = 4,
              start_bias: float = 0.0, min_screens: int = MIN_SCREENS) -> dict[str, Any]:
    """Rounds of `play(bias)` (the run rows of a block played with that bias) until a round's Act 1 skip rate is within `tol` of
    `target`, or `max_rounds`. The result: the bias that came closest in a played round (`bias`), whether that round was within the
    tolerance (`landed`), and the rounds."""
    rounds: list[dict[str, Any]] = []
    bias = float(start_bias)
    for index in range(max_rounds):
        rows = play(bias)
        rate, screens = skip_rate(rows)
        gaps = act1_gaps(rows)
        rec: dict[str, Any] = {"round": index, "bias": bias, "skip_rate": rate, "screens": screens, "gaps": len(gaps), "runs": len(rows),
                               "median_gap": round(statistics.median(gaps), 3) if gaps else None}
        rounds.append(rec)
        if rate is not None and screens >= min_screens and abs(rate - target) <= tol:
            rec["landed"] = True
            break
        rec["landed"] = False
        if not gaps:
            rec["stop"] = "the rows carry no skip gaps (is the agent's macro a skipcal pick?)"
            break
        nxt = bias_for(gaps, target)
        rec["open_loop_rate_at_next"] = round(rate_at(gaps, nxt), 4)
        rec["next_bias"] = nxt
        if abs(nxt - bias) < 0.0005:
            rec["stop"] = "the next bias is this one: more rounds would play the same block"
            break
        bias = nxt
    played = [r for r in rounds if r["skip_rate"] is not None]
    best = min(played, key=lambda r: abs(r["skip_rate"] - target)) if played else None
    return {"target": target, "tolerance": tol, "bias": best["bias"] if best else None, "landed": bool(best and best["landed"]),
            "skip_rate": best["skip_rate"] if best else None, "rounds": rounds}


def seed_agent(agent_dir: str | Path, bias: float, keep_search: bool):
    """The agent with its chooser at `bias` (a skipcal macro on the folder's human prior, lambda and run judgement), and, unless
    `keep_search`, with the plain search (no stakes schedule, no hold)."""
    import gen_loop  # noqa: PLC0415

    base = gen_loop.load_agent(agent_dir)
    if not macro_recipe.is_blended(base.macro):
        raise ValueError(f"{agent_dir}: the macro {base.macro!r} is not a blended chooser; build a candidate with --human-prior first")
    lam, sha8 = macro_recipe.parse_macro(base.macro)
    macro = macro_recipe.macro_name(lam, sha8, macro_recipe.PICK_SKIPCAL, bias)
    plain = {} if keep_search else {"stakes": None, "hold": None}
    return dataclasses.replace(base, macro=macro, **plain)


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--agent-dir", required=True, help="a built candidate whose macro is a blended chooser (its prior, lambda and judgement are used)")
    ap.add_argument("--out", required=True, help="a folder: one sub-folder a round, and calibration.json")
    ap.add_argument("--lib", required=True, help="the native library's folder (STS2_LIB_PATH)")
    ap.add_argument("--runs", type=int, default=50)
    ap.add_argument("--procs", type=int, default=2, help="processes (one emulator each); at most 2 beside the lab's jobs")
    ap.add_argument("--config", default="a3-v1", help="the search config the runs play (default a3-v1)")
    ap.add_argument("--keep-search", action="store_true", help="play the agent's own stakes schedule and hold instead of the plain search")
    ap.add_argument("--seeds-name", default="SKIPCAL-2026-11-06", help="the quick check's seed block name (gen0.e_block): never an E, S or N block")
    ap.add_argument("--target", type=float, default=HUMAN_ACT1_SKIP)
    ap.add_argument("--tol", type=float, default=TOLERANCE)
    ap.add_argument("--max-rounds", type=int, default=4)
    ap.add_argument("--start-bias", type=float, default=0.0)
    ap.add_argument("--quiet-tag", default="skipcal", help="the QUIET.lock line this heavy job holds (never edit others'); '' for none")
    a = ap.parse_args(argv)
    if a.procs > 2:
        ap.error("--procs is at most 2")
    if a.seeds_name.startswith(("E", "S", "N")):
        ap.error("--seeds-name must not look like an E, S or N block")
    os.environ["STS2_LIB_PATH"] = a.lib
    os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")
    import gen0  # noqa: PLC0415
    import quiet_lock  # noqa: PLC0415

    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    seeds = gen0.e_block(a.seeds_name, a.runs)

    def play(bias: float) -> list[dict]:
        import gen_loop  # noqa: PLC0415

        agent = seed_agent(a.agent_dir, bias, a.keep_search)
        arm = out / f"round-bias{macro_recipe.format_bias(bias)}"
        if arm.exists() and (arm / "runs.jsonl").is_file() and len(gen_loop.arm_rows(arm)) >= len(seeds):
            return gen_loop.arm_rows(arm)  # a rerun keeps a finished round
        gen0.run_block(seeds, a.config, a.procs, arm, agent=agent)
        return gen_loop.arm_rows(arm)

    lock = quiet_lock.QuietLock(Path(QUIET_FILE), a.quiet_tag, wait_h=0.5) if a.quiet_tag else None
    if lock is not None:
        if not lock.acquire():
            print(f"QUIET.lock stays taken by {lock.blocked_by()}: not started", file=sys.stderr)
            return 2
        lock.start_heartbeat()
    try:
        result = calibrate(play, target=a.target, tol=a.tol, max_rounds=a.max_rounds, start_bias=a.start_bias)
    finally:
        if lock is not None:
            lock.release()
    result["runs_per_round"] = a.runs
    result["seeds_name"] = a.seeds_name
    result["search"] = "the agent's own" if a.keep_search else a.config
    (out / "calibration.json").write_text(json.dumps(result, indent=1), encoding="utf-8")
    print(json.dumps({k: result[k] for k in ("bias", "landed", "skip_rate", "target")}))
    return 0 if result["landed"] else 1


if __name__ == "__main__":
    sys.exit(main())
