"""The always-on fight-gym loop (D40, gym build item L1): generation, a trainer on the card, a promotion gate, the board.

    python rl/gym_loop.py starts  --root ROOT [--runs-per-night 250] [--procs 6]    the training start set (resumable)
    python rl/gym_loop.py weights --root ROOT [--procs 6]                           the start mix's kind weights (a3-v1 pass)
    python rl/gym_loop.py run     --root ROOT [--player-config a3-v1] [--procs 6]   the daemon (detached: see GYM_LOOP.md)
        [--procs 0 --gate-procs 4]   generation off: only the trainer, the pre-filter and the gate run (D65 addendum 4)
        [--trainer-extra-rows-root PATH ... --trainer-extra-window N --trainer-policy-min-grade N]   the plain runs' fight rows
    python rl/gym_loop.py gate    --root ROOT --candidate DIR                      one promotion gate, by hand
    python rl/gym_loop.py gate5-set --root ROOT [--seed N] [--rebuild]             rule v5's own ruler (gate5/), once
    python rl/gym_loop.py stop    --root ROOT                                       stop the daemon (its STOP file, then its PID)
    python rl/gym_loop.py status  --root ROOT

The daemon does, in turn, while the lab's locks allow it (`why_blocked`):

1. **Generation.** A segment of fights (about `--segment-min` minutes of six processes) from the start mix: every fight kind of
   Acts 1 and 2 from every finished night's emulator arm (`nights_available`: the newest `--max-nights`, `--runs-per-night` runs of
   each), sampled by kind in proportion to where the runs lose HP (`kind_weights`), then by source and by regret within the kind
   (`practice_mix`: a share of each kind's draws from fights rebuilt from human runs, `rl/human_starts.py`, the rest from the nights'
   AI runs and from the positions three decisions before the AI's own losses; within a kind and side the rank of the start's regret,
   the teacher's root value at the opening less the outcome, mixed with a staleness term; the exam's `exam/exclude_runs.json` is
   never drawn from; the segment's log line says what was drawn; with `--teacher-store` a fourth source, the run teacher's own Act 1 and 2
   elite and boss fights, `rl/teacher_starts.py`, `teacher_share` in mix.json). The mix follows
   the nights: between steps the daemon rebuilds the start set when a night has finished since it was built (`starts_stale`; only the
   new night's runs are replayed), and recomputes the kind weights when the set or the incumbent has changed (`weights_stale`; the
   incumbent's weight games are kept and only topped up). The player is the newest checkpoint (the incumbent before the first
   one) under the teacher search (`--teacher`, x16), played by P1's lockstep processes (`gym_lockstep.run_lockstep`) with the
   prior on the card when GPU_OK allows, else on the CPU. Every decision is recorded (`gen_game_steps`): the position's tokens as
   the prior reads them, the root's visit distribution and mean values (`gen_loop.root_targets`, the self-play layout), the legal
   mask, and the exact solver's grade where it finishes (`gym_lockstep.Grader`, expected solve only). Each fight's samples go to
   one file; after the segment they are compacted into one shard of the replay buffer, with the fight's outcome as the value label
   (end HP, 0 for a death; a capped fight has no value label), the player version that made it, its kind, its L0 and its split (by
   run seed).
2. **Training** runs beside it in its own process on the card (`rl/gym_loop_train.py`, the GPU venv): over the buffer's newest
   window, from the incumbent's weights, a checkpoint every 30 minutes, at most `--max-epochs` epochs over unchanged data (then it
   idles and logs the idle time).
3. **The gate.** Only checkpoints that pass the pre-filter (D49, `prefilter_pick`) enter it: a checkpoint's held-out policy
   cross-entropy must be below the incumbent's on the same rows, and below that of every checkpoint already rejected against this
   incumbent; at least one gate every six hours regardless (the best of the skipped ones). The rule below is untouched.
   A checkpoint not yet scored is played on the fight board's 200 Act-1 boss starts with the board's seeds, under the
   player config (`--player-config`: a3-v1, or the judge-equipped config once it lands), paired with the incumbent on the same games
   (the incumbent's games are played once per incumbent, engine and config, and kept). Stage 1 uses a pair of seed indices the
   incumbent was not picked on, one pair per promotion, with the incumbent's games on it taken from its own promotion gate (D43).
   `--gate-every-min N` (D60 item 5; 0, today's behaviour, by default) offers the gate only a checkpoint made N minutes after the one it last
   played, so one in four at 120 against the trainer's 30-minute checkpoints.
   The rule (`PROMOTION_RULE`) was fixed here before the first checkpoint and amended twice (v2, 2026-11-01; v3, 2026-11-11, D57 item 2:
   stage 1 stops only a candidate clearly worse, stage 2 promotes alone by v2's test or when the candidate is not worse and the pooled
   paired difference of the last five stage-2 candidates of its incumbent is positive, at most one promotion in six hours;
   `rl/gym_gate.py`); `decide` applies it. A promoted checkpoint becomes "Fight player, version N": its games get a fight-board row,
   committed to Spireline main (`board_commit`).
   **The human bar** (D55 addendum 11, `--human-bar promotion|on|off`, promotion): a promotion (in mode `on`, any stage-2 verdict) is
   followed by the candidate's 400 human-stratum fight starts of the fight exam (`rl/fight_exam.py`, four draws, the exam's own
   configuration, the CPU prior), played by the gate's processes; the candidate's, the incumbent's and the human's survival and HP lost on
   those starts go into the verdict (`human`), the loop log and `gate/verdicts.jsonl`. It is report-only: `PROMOTION_RULE` and `decide`
   never read it, and a play a stop cuts short is recorded as `human: "stopped"`.
   **The plateau alarm and the clock** (D57 addendum 2): after five days or twenty-five gated candidates without a promotion the daemon
   writes LAB/alerts/gym-plateau.txt (`gym_gate.plateau_update`); every wait it makes is logged with its reason and duration and timed
   by reason in `games/clock.jsonl`, with a daily line at 00:00Z (`rl/gym_clock.py`).

Between steps the daemon drops its QUIET.lock line and waits `--yield-s` so a waiting gate can take the lock; it never holds the
line through the night (03:10-03:45Z and while LAB/nightly/active exists), and starts nothing while main.lock, LAB/PAUSE or LAB/HALT
exist or another live QUIET.lock line stands. A running step is stopped at once by the night, PAUSE, HALT, its STOP file or another
QUIET.lock line; main.lock lets it finish (as the lab's guards treat a running job). Every process is BelowNormal and in the daemon's
Job Object, so nothing it started outlives it; it is stopped by its STOP file or by its own PID. Everything is resumable: the buffer,
the checkpoints, the gate's games and the promotion state are on disk, and a restart carries on from them.

Lab code: nothing here changes the engine, the search or the solver. The human corpus's exam fifth (D38 addendum 1) is never a
start: `rl/human_starts.py` leaves it out of the set it writes, and `practice_mix` leaves out whatever the fight exam lists.
"""

from __future__ import annotations

import argparse
import collections
import contextlib
import datetime as dt
import hashlib
import json
import math
import os
import random
import re
import shutil
import subprocess
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
import gym_clock as gclock  # noqa: E402
import gym_gate as gg  # noqa: E402
import gym_wiring as gw  # noqa: E402
import practice_mix as pmix  # noqa: E402

# Nothing here loads the emulator library at import (gym_lockstep's rule): a spawned process selects its engine first.

TAG = "gym-loop"  # QUIET.lock and GPU_OK line of the daemon
TRAIN_TAG = "gym-loop-train"  # GPU_OK line of the trainer
MY_TAGS = (TAG, TRAIN_TAG)
REF = fg.REF
LAB = REF / "LAB"
PAUSE, HALT = LAB / "PAUSE", LAB / "HALT"
GPU_OK = REF / "GPU_OK"
GPU_OFF = LAB / "GPU_OFF"
GPU_PYTHON = os.environ.get("SPIRELINE_GPU_PYTHON", "D:/spireline/venv-gpu/Scripts/python.exe")
DEFAULT_ROOT = REF / "gym-loop"
STATUS_DIR = Path(os.environ.get("GYM_LOOP_STATUS_DIR", "D:/spireline/logs/gym-loop"))
BOARD_DIR = REF / "lab" / "results" / "fight-gym" / "2026-10-29-m2"  # the fight board's starts and its three standing arms
INCUMBENT0 = REF / "agent" / "gen7" / "candidate-r1"  # M2's agent: version 0, the board's "Usual thinking"
# The start set: every finished night's emulator arm (`nights_available`; until 2026-11-02 the four nights 10-26 to 10-30 at 50 runs
# each, 1,478 starts). D44 step E: the trainer waited for fresh practice on a thin set, so the set takes each night as it comes.
RUNS_PER_NIGHT = 250  # runs of each night's arm replayed into the set (a night holds 500 or 2,000)
MAX_NIGHTS = 14  # the newest nights the set takes: a window, so it stays bounded (about 2,100 starts and 28 MB a night)
_NIGHT_DATE = re.compile(r"^\d{4}-\d{2}-\d{2}(?:T\d{4})?$")  # a night folder: the date, or the date and start time of a block (rl/nightly_real.py BLOCK_RE)
ENGINE = "core2"
NS_GEN, NS_WEIGHTS = "L", "W"  # generation's and the weights pass's seed namespaces; the gym's (and the gate's) is "g"
KINDS = ("act1-hallway", "act1-elite", "act1-boss", "act2-hallway", "act2-elite", "act2-boss")
NODE_KIND = {"M": "hallway", "E": "elite", "B": "boss"}
NODE_TYPE = {"hallway": 1, "elite": fg.NODE_ELITE, "boss": fg.NODE_BOSS}  # run_train.vocab's node types
LAST_ACT2_INDEX = 31  # the run row's nodes: index i is floor i + 2; Acts 1 and 2 are indices 0..31 (floor 33, the Act-2 boss)
VAL_SHARE = 0.1  # run seeds whose fights are the trainer's held-out set (split by run seed, never by start: no deck in both)
MAX_ACTIONS = 32
QUIET_STALE_H = 2.0
# Other lanes' measurement lines the daemon plays beside (D58 addendum 3: the loop is not held by a builder's measurement);
# `--quiet-allow` on `run`. A judgement's lab-train line and every other tag still hold it.
QUIET_ALLOW: tuple[str, ...] = ("run-teacher", "judge-corpus", "judge-train")
NIGHT_WINDOW = fg.NIGHT_WINDOW
# Own seconds per x16 game by kind before any is measured (M2 and P1's numbers; a hallway fight is a short elite's).
MEAN_X16_S = {"act1-hallway": 12.0, "act1-elite": 25.0, "act1-boss": 74.0, "act2-hallway": 14.0, "act2-elite": 16.0,
              "act2-boss": 95.0, "act3-boss": 120.0}  # act3-boss: the Act 3 stratum (practice_mix, off by default); no measure yet

# The promotion rule. v1 was fixed 2026-10-30 before the first checkpoint, v2 on 2026-11-01 (D43), v3 on 2026-11-11 (D57 item 2).
# Both arms play the same games: the board's 200 Act-1 boss starts, game j of each with the gym's seeds (namespace "g"), the same
# search config; only the network differs. A capped game is lost. Stage 1 plays one pair of seed indices (`stage1_seeds` of them, 400
# games), the pair after the one the incumbent was picked on (one pair per promotion, wrapping: 0-1, 2-3, ..., `stage1_js`,
# `next_gate_pair`; its incumbent games come from the incumbent's own promotion gate when they are there, `reuse_incumbent_games`).
# Stage 2 plays j = 0..seeds-1 (the board's 2,000 games). v2 stopped at stage 1 below +1.0 points and promoted only when the exact
# one-sided McNemar test on the discordant pairs gave p < 0.05 with the paired mean L0 difference not negative; between 2026-10-31
# and 11-09 it refused 175 candidates at +0.0 to +1.1 points (SE 0.65), a gain it cannot show on 2,000 pairs. v3 (D57: "not worse, and
# the trend across candidates positive") keeps v2's test as path (a) and adds path (b); stage 1 becomes a screen of the clearly worse.
# `rl/gym_gate.py` has the statistics and their reasons: start-clustered standard errors; the screen rejects when the one-sided 95%
# UPPER bound of the stage-1 difference is under `screen.points`; (b) needs the candidate's 95% lower bound above `pool.not_worse_points`
# and the pooled paired difference of the last `pool.k` stage-2 candidates of this incumbent and rule version (at least `pool.min` of
# them; the candidates averaged per start, one paired test over starts: `gym_gate.pooled`) positive at one-sided p < `pool.alpha`; at most
# one promotion in `guard_h` hours. Nothing tunes on these games: the trainer
# never sees the board's starts or their runs.
PROMOTION_RULE_V2 = {"name": "gym-loop-promote-v2", "fixed": "2026-11-01 (D43): stage 1's pair rotates; v1 fixed 2026-10-30",
                     "starts": 200, "cat": "act1-boss", "stage1_seeds": 2, "stage1_rotates": True, "futility_points": 1.0,
                     "seeds": 10, "alpha": 0.05, "l0_not_worse": True, "capped": "lost", "ns": "g"}
PROMOTION_RULE = {"name": "gym-loop-promote-v3", "fixed": "2026-11-11 (D57 item 2): a screen at stage 1, a pooled trend at stage 2",
                  "starts": 200, "cat": "act1-boss", "stage1_seeds": 2, "stage1_rotates": True, "seeds": 10, "capped": "lost", "ns": "g",
                  "screen": {"bound": "upper", "z": gg.Z95, "points": -2.0},
                  "single": {"alpha": 0.05, "l0_not_worse": True, "min_points": 0.0},
                  "pool": {"k": 5, "min": 3, "alpha": 0.10, "not_worse_points": -1.0, "z_lower": gg.Z95},
                  "guard_h": 6.0}
# Rule v4 (D57 addendum 9, 2026-11-20; `--rule v4|v3`, v4 the default, `--rule v3` the undo). The ruler is the fight exam's human stratum
# (`exam/starts-human.jsonl`, 400 starts of Acts 1 and 2) under the exam's own configuration (`fight_exam.DEFAULT_CONFIG`, the CPU prior,
# as `human_bar` played it), paired by (start, draw j) with the incumbent on the same seeds; survival = won and not capped. The statistics
# are v3's (`gym_gate.cluster_mean_se`). Stage 1 plays draws 0-1 and rejects when the 95% upper bound of the survival gain is under
# `screen.points`; stage 2 draws 0-7 promotes at `promote.points` with the 90% lower bound above zero and HP lost a fight not worse by
# more than `promote.hp_worse`; a gain of `near_points` or more that misses the bar goes to stage 3, draws 0-15, the same test, else reject.
# The incumbent's games are kept in `exam/games-<incumbent>.jsonl` and reused after an identity check (`identity.n` stored games replayed
# on a new engine: outcome, end HP and move hash must match). The Act 3 guard (`act3_guard`): from `after_h` hours after the first practice
# segment that drew Act 3 starts, a would-be promotion also plays one draw of a fixed subsample of the Act 3 exam cells and fails when the
# 95% upper bound of the survival difference is under `points`. The pooled-trend path of v3 is retired for v4 (`gym_gate.pooled` stays for v3).
PROMOTION_RULE_V4 = {"name": "gym-loop-promote-v4", "fixed": "2026-11-20 (D57 addendum 9): the exam's human stratum, paired draws, stages of 2, 8 and 16 draws",
                     "config": "a3-v1", "prior": "cpu", "starts": 400, "capped": "lost", "stage_draws": (2, 8, 16),
                     "screen": {"bound": "upper", "z": gg.Z95, "points": -1.0},
                     "promote": {"points": 0.75, "z_lower": gg.Z90, "hp_worse": 0.5}, "near_points": 0.5,
                     "identity": {"n": 24, "seed": "gym-loop-identity"},
                     "act3_guard": {"starts": 120, "quotas": {"80": 80}, "seed": 20261120, "draws": 1, "points": -3.0, "z": gg.Z95,
                                    "after_h": 24.0, "when": "promote"},
                     "guard_h": 6.0}
# Rule v5 (D68 addendum 5 decision 4, 2026-11-21; `--rule v5|v4|v3`, v5 the default; v4 and v3 the undo paths). Its ruler is its own
# (`gate5/`, built once by `gate5-set`; its runs go on the exam's exclusion list so the practice never draws them): part `h`, 500 human starts
# of Acts 1 and 2 that the fight exam did not take (the exam's 400 are held out again), and part `a3`, 414 starts at the two Act 3 bosses
# drawn a second time by lane 5's design (the exam's own 414 stay held out; `set.act3_source: exam` reads them instead). Within a part a cell
# counts by our runs' deaths there per 100 runs (the real block of generation 20, 2026-11-20, 1,994 runs); the stratified estimate
# (`gym_gate.strat_read`). Two routes (`gym_gate.v5_decide`): F, part h gains at least +0.75 points with its 97.5% lower bound above 0 and HP
# lost not worse by 0.5 while part a3 is not clearly worse (95% upper bound under -2.0); A3, part a3 gains at least +1.5 with the same bound
# and HP test while part h is not clearly worse (95% upper bound under -1.0). Stage 1 (h 2 draws, a3 1) rejects a clearly worse part; stages
# 2 (8, 4) and 3 (16, 8) promote by a route; stage 2 goes on when a part gains its `near` points. Draws are numbered by era (the promotions
# under v5): era e plays h draws 16e.. and a3 draws 8e.., so a new incumbent's bar is played fresh, never its selection games.
GATE5_WEIGHTS = {"h": {"act1-hallway": 2.5, "act1-elite": 2.3, "act1-boss": 22.8, "act2-hallway": 10.1, "act2-elite": 10.2, "act2-boss": 22.1},
                 "a3": {"act3-boss1-73": 4.71, "act3-boss1-78": 4.81, "act3-boss1-80": 4.86, "act3-boss2-73": 1.29, "act3-boss2-78": 1.29,
                        "act3-boss2-80": 1.29}}
GATE5_H_QUOTA = {**{f"act1-boss-{b}": 28 for b in (74, 77, 79, 82, 83, 84)}, **{f"act2-boss-{b}": 54 for b in (75, 76, 81)},
                 "act2-elite": 85, "act2-hallway": 40, "act1-elite": 25, "act1-hallway": 20}
PROMOTION_RULE_V5 = {"name": "gym-loop-promote-v5", "fixed": "2026-11-21 (D68 addendum 5 decision 4): its own ruler of two parts (Acts 1-2 "
                     "human starts outside the exam; the Act 3 bosses), deaths-weighted cells, two routes at 97.5%, fresh draws per era",
                     "config": "a3-v1", "prior": "cpu", "capped": "lost",
                     "parts": {"h": {"stage_draws": (2, 8, 16), "era_draws": 16, "worse": -1.0, "near": 0.5, "weights": GATE5_WEIGHTS["h"]},
                               "a3": {"stage_draws": (1, 4, 8), "era_draws": 8, "worse": -2.0, "near": 1.0, "weights": GATE5_WEIGHTS["a3"]}},
                     "routes": {"F": {"part": "h", "points": 0.75, "z": gg.Z975, "hp_worse": 0.5},
                                "A3": {"part": "a3", "points": 1.5, "z": gg.Z975, "hp_worse": 0.5}},
                     "z_worse": gg.Z95,
                     "identity": {"n": 24, "seed": "gym-loop-identity"},
                     "set": {"seed": 20261121, "h_quota": GATE5_H_QUOTA, "act3_source": "twin",
                             "a3_human_quota": {"act3-boss1-80": 80, "act3-boss2-80": 40, "act3-boss1-78": 32, "act3-boss1-73": 32,
                                                "act3-boss2-78": 20, "act3-boss2-73": 20},
                             "a3_own_quota": {"act3-boss1-80": 60, "act3-boss2-80": 30, "act3-boss1-78": 30, "act3-boss1-73": 30,
                                              "act3-boss2-78": 20, "act3-boss2-73": 20}},
                     "guard_h": 6.0}
RULES = {"v3": PROMOTION_RULE, "v4": PROMOTION_RULE_V4, "v5": PROMOTION_RULE_V5}


def active_rule(a) -> dict:
    """The rule the daemon (or `gate`) runs under: `--rule` (v5 by default on the command line); an argument set with no `rule` is v3, as
    before the flag."""
    return RULES[getattr(a, "rule", None) or "v3"]


# The gate's pre-filter (D49 item 3, 2026-11-03; it decides who enters the gate, never who is promoted). Every checkpoint's train.json
# carries its held-out policy cross-entropy beside the incumbent's on the same rows (the trainer scores both in one call). The
# policy is what the gate's player reads (the prior), so the policy figure is the one compared; "gap" is the incumbent's minus the
# checkpoint's (positive: the checkpoint is better). Measured when it was written: all 78 checkpoints gated against v2 (c0020 to
# c0100) had a positive gap, 0.0028 to 0.0129 nats from c0049 on (before it the trainer still compared with an older version), and none was
# promoted, so "below the incumbent's" alone skips nothing; mode "best" also asks the gap to exceed that of every checkpoint
# already gated and rejected against this incumbent.
# `margin` is nats the gap must exceed; `floor_h` hours after which the best-gap skipped checkpoint of the window is gated.
PREFILTER = {"name": "gym-loop-prefilter-v1", "fixed": "2026-11-03 (D49)", "metric": "val_policy_ce", "mode": "best",
             "margin": 0.0, "floor_h": 6.0}

_HOME = re.compile(r"(?i)[a-z]:[\\/]+users[\\/]+[^\\/\s'\"]+")


def clean(text: str) -> str:
    """No user paths in logs or results: a Windows home folder becomes `~`."""
    return _HOME.sub("~", str(text))


def utc() -> dt.datetime:
    return dt.datetime.now(dt.timezone.utc)


def stamp() -> str:
    return utc().strftime("%Y-%m-%dT%H:%M:%SZ")


def read_json(p: Path, default=None):
    try:
        return json.loads(Path(p).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


def write_json(p: Path, obj, patience_s: float = 10.0) -> None:
    """Atomic: written beside, then renamed over. Windows refuses the rename while a reader has the target open (the lab's runner
    reads the daemon's status file every few seconds; this ended the daemon at 07:18Z on 2026-11-03, rc=1, in a `waiting` write):
    it is tried again for `patience_s` seconds before the error is raised."""
    p = Path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    tmp = p.with_name(p.name + f".tmp{os.getpid()}-{threading.get_ident()}")
    tmp.write_text(json.dumps(obj, indent=1), encoding="utf-8", newline="\n")
    end = time.monotonic() + patience_s
    while True:
        try:
            os.replace(tmp, p)
            return
        except PermissionError:
            if time.monotonic() >= end:
                tmp.unlink(missing_ok=True)
                raise
            time.sleep(0.2)


def sha256_file(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_val_seed(run_seed: str) -> bool:
    """The trainer's held-out split, by run seed: a fixed tenth of seeds."""
    return int(hashlib.sha256(f"gym-loop-val:{run_seed}".encode()).hexdigest()[:8], 16) / 0xFFFFFFFF < VAL_SHARE


def val_split_id() -> str:
    """Names the held-out split (the rule and its share); a held-out figure is the same measure only under the same id."""
    return f"gym-loop-val:{VAL_SHARE}"


# ---------------------------------------------------------------------------------------------------------- the locks


def _lines(p: Path) -> list[str]:
    try:
        return [x.strip() for x in p.read_text(encoding="utf-8", errors="replace").splitlines() if x.strip()]
    except FileNotFoundError:
        return []
    except OSError:
        return []


def other_quiet(lines: list[str] | None = None, *, allow: tuple[str, ...] = ()) -> str | None:
    """The tag of another live QUIET.lock line (not ours, not w3-blocks, not older than 2 h, not in `allow`), or None."""
    for x in _lines(fg.QUIET) if lines is None else lines:
        tag = x.split()[0]
        if tag in MY_TAGS or tag == "w3-blocks" or tag in allow:  # our builds and gates (gym-loop-*) stop the daemon too
            continue
        if fg._age_h(x) < QUIET_STALE_H:  # noqa: SLF001
            return tag
    return None


def in_night_window(now: dt.datetime | None = None) -> bool:
    t = (now or utc()).time()
    return NIGHT_WINDOW[0] <= t < NIGHT_WINDOW[1]


def night_on() -> bool:
    return fg.ACTIVE.exists() or in_night_window() or any(x.startswith("lab-nightly") for x in _lines(fg.QUIET))


def why_blocked(root: Path, *, starting: bool) -> str | None:
    """Why the daemon must not run heavy work now (None: it may). `starting`: about to start a step (main.lock stops a start only:
    a running step carries on through a landing, as the lab's guards let a running job)."""
    if (root / "STOP").exists():
        return "stop"
    if HALT.exists():
        return "halt"
    if fg.ACTIVE.exists():
        return "nightly-active"
    if in_night_window():
        return "night-window"
    if PAUSE.exists():
        return "pause"
    if starting and fg.MAIN_LOCK.exists():
        return "main.lock"
    tag = other_quiet(allow=QUIET_ALLOW)
    return f"quiet:{tag}" if tag else None


def hold_quiet(on: bool) -> None:
    """Hold (or drop) the daemon's QUIET.lock line, with its Windows PID; other lines are kept as they are."""
    keep = [x for x in _lines(fg.QUIET) if x.split()[0] != TAG]
    if on:
        keep.append(f"{TAG} {stamp()} pid={os.getpid()}")
    if keep:
        fg.QUIET.write_text("".join(x + "\n" for x in keep), encoding="utf-8", newline="\n")
    else:
        fg.QUIET.unlink(missing_ok=True)


def gpu_allowed(lines: list[str] | None = None) -> bool:
    """The card may be used: GPU_OK is absent or empty, or its first line is one of ours."""
    lines = _lines(GPU_OK) if lines is None else lines
    return not lines or lines[0] in MY_TAGS


def take_gpu(tag: str) -> bool:
    """Put `tag` in GPU_OK beside our other line, only while the first line is absent or ours."""
    lines = _lines(GPU_OK)
    if not gpu_allowed(lines):
        return False
    if tag not in lines:
        lead = 0
        while lead < len(lines) and lines[lead] in MY_TAGS:
            lead += 1
        lines.insert(lead, tag)
        GPU_OK.write_text("".join(x + "\n" for x in lines), encoding="utf-8", newline="\n")
    return True


def drop_gpu(tag: str) -> None:
    if not GPU_OK.exists():
        return
    lines = [x for x in _lines(GPU_OK) if x != tag]
    if lines:
        GPU_OK.write_text("".join(x + "\n" for x in lines), encoding="utf-8", newline="\n")
    else:
        GPU_OK.unlink(missing_ok=True)


# ---------------------------------------------------------------------------------------------------------- the state


class Loop:
    """The loop's folder: the start set, the buffer, the checkpoints, the players, the gate's games, the state and the log."""

    def __init__(self, root: Path | str) -> None:
        self.root = Path(root)
        self.starts = self.root / "starts"
        self.incoming = self.root / "buffer" / "incoming"
        self.shards = self.root / "buffer" / "shards"
        self.checkpoints = self.root / "checkpoints"
        self.players = self.root / "players"
        self.gates = self.root / "gate"
        self.games = self.root / "games"
        self.board = self.root / "board"
        self.trainer = self.root / "trainer"
        for d in (self.starts, self.incoming, self.shards, self.checkpoints, self.players, self.gates, self.games, self.board,
                  self.trainer):
            d.mkdir(parents=True, exist_ok=True)
        self.state_path = self.root / "state.json"
        self.state = read_json(self.state_path, {}) or {}
        self.state.setdefault("segment", 0)
        self.state.setdefault("next_j", {})
        self.state.setdefault("last_seg", {})  # start id -> the segment it was last drawn in (the sampler's staleness term)
        self.mix: dict = {}  # what the last `sample_tasks` drew (kind and source counts, the regret quartiles)
        self.clock = None  # the daemon's `gym_clock.TimeLedger` once it runs (waits and plays by reason); None: nothing is timed
        self.state.setdefault("kind_s", {})
        self.state.setdefault("gated", {})
        self.state.setdefault("versions", {"v0": {"dir": str(INCUMBENT0), "source": "agent gen7/candidate-r1 (M2's)", "n": 0}})
        self.state.setdefault("incumbent", "v0")
        self.state.setdefault("promotions", [])
        # Which pair of seed indices stage 1 plays (see PROMOTION_RULE): 0 is j = 0..1. A state from before the rotation had
        # promoted only on pair 0, so its incumbent starts on the next one.
        self.state.setdefault("gate_pair", 1 if self.state["promotions"] else 0)
        self._lock = threading.Lock()

    def save(self) -> None:
        with self._lock:
            write_json(self.state_path, self.state)

    def log(self, msg: str) -> None:
        line = f"{stamp()} {clean(msg)}\n"
        with self._lock, open(self.root / "loop.log", "a", encoding="utf-8") as fh:
            fh.write(line)

    def span(self, kind: str, what: str, log: bool = False):
        """A timed interval of the daemon's day (`gym_clock`): a context manager that times nothing when no clock runs."""
        return self.clock.span(kind, what, log=log) if self.clock is not None else contextlib.nullcontext()

    def player_dir(self, version: str) -> Path:
        return Path(self.state["versions"][version]["dir"])

    def incumbent(self) -> tuple[str, Path]:
        v = self.state["incumbent"]
        return v, self.player_dir(v)

    def gate_pair(self) -> int:
        """Stage 1's pair (an index, 0 up), kept inside the rule's pairs."""
        return int(self.state["gate_pair"]) % stage1_pairs(PROMOTION_RULE)

    def newest_checkpoint(self) -> tuple[str, Path] | None:
        done = sorted(p for p in self.checkpoints.glob("c[0-9][0-9][0-9][0-9]") if (p / "model.json").is_file()
                      and (p / "combat_policy.pt").is_file())
        return (done[-1].name, done[-1]) if done else None

    def generation_player(self) -> tuple[str, Path]:
        """The newest checkpoint, or the incumbent before the first: players reload between fights (between segments here)."""
        return self.newest_checkpoint() or self.incumbent()


# ---------------------------------------------------------------------------------------------------------- the start set


def board_run_seeds() -> set[str]:
    """Every run seed behind M2's starts (the board's 200 Act-1 bosses and their runs' Act-2 fights): never a training start."""
    return {s["seed"] for s in fg.read_jsonl(BOARD_DIR / "starts.jsonl")}


def fight_floors(run: dict) -> list[dict]:
    """The fights a night's run entered in Acts 1 and 2 (hallway, elite, boss; not events), up to the floor it reached."""
    nodes, floor = run.get("nodes") or "", int(run["floor"])
    out = []
    for i, ch in enumerate(nodes[: LAST_ACT2_INDEX + 1]):
        if ch not in NODE_KIND or i + 2 > floor:
            continue
        act = 1 if i <= fg.ACT1_BOSS_INDEX else 2
        out.append({"floor": i + 2, "act": act, "kind": NODE_KIND[ch], "cat": f"act{act}-{NODE_KIND[ch]}"})
    return out


def fights_per_run(runs: list[dict]) -> dict[str, float]:
    """Mean fights of each kind a run enters (a run that dies early enters fewer of the later ones)."""
    c: Counter = Counter()
    for r in runs:
        c.update(f["cat"] for f in fight_floors(r))
    n = max(1, len(runs))
    return {k: c[k] / n for k in KINDS}


def pick_runs(runs: list[dict], n: int, seed: int, exclude: set[str]) -> list[dict]:
    ok = [r for r in fg._eligible(runs) if r["seed"] not in exclude]  # noqa: SLF001
    return random.Random(seed).sample(ok, min(n, len(ok)))


def nights_available(lab: Path | None = None, max_nights: int = MAX_NIGHTS) -> tuple[str, ...]:
    """Every finished night's emulator arm, oldest first: `LAB/nightly/<date>/<agent>-emu` with its summary.json (written at the
    arm's end) and runs.jsonl; the newest `max_nights` of them (0: all)."""
    base = (LAB if lab is None else Path(lab)) / "nightly"
    found = sorted(f"{d.name}/{a.name}" for d in base.glob("*") if d.is_dir() and _NIGHT_DATE.match(d.name)
                   for a in d.glob("*-emu") if (a / "summary.json").is_file() and (a / "runs.jsonl").is_file())
    return tuple(found[-max_nights:]) if max_nights > 0 else tuple(found)


def night_seed(seed: int, label: str) -> int:
    """A night's own draw of runs, fixed by its label: adding or dropping other nights never changes a night's picks."""
    return seed + int(hashlib.sha256(label.encode()).hexdigest()[:8], 16)


def start_spec(nights: tuple[str, ...], runs_per_night: int) -> dict:
    return {"nights": [n.replace("-emu", "") for n in nights], "runs_per_night": int(runs_per_night)}


def starts_stale(loop: Loop, nights: tuple[str, ...], runs_per_night: int) -> str | None:
    """Why the start set must be built again for these nights and runs a night (None: it is current). A set written before the
    nights were found by `nights_available` is read by its meta's nights and runs a night."""
    meta = read_json(loop.starts / "meta.json", None)
    if not meta or not (loop.starts / "starts.jsonl").is_file():
        return "no start set"
    have = meta.get("spec") or {"nights": sorted(meta.get("nights") or {}), "runs_per_night": meta.get("runs_per_night")}
    want = start_spec(nights, runs_per_night)
    if have.get("runs_per_night") != want["runs_per_night"]:
        return f"runs per night {have.get('runs_per_night')} to {want['runs_per_night']}"
    old = list(have.get("nights") or [])
    if old != want["nights"]:
        new, gone = [n for n in want["nights"] if n not in old], [n for n in old if n not in want["nights"]]
        return f"nights: {len(new)} new, {len(gone)} gone"
    return None


def iter_jsonl(p: Path):
    """`fight_gym.read_jsonl` one line at a time (a replay file holds every start's document: never all of it in memory)."""
    if not p.is_file():
        return
    with open(p, encoding="utf-8") as fh:
        for x in fh:
            if x.strip():
                try:
                    yield json.loads(x)
                except ValueError:
                    pass  # a line cut by a stop: its run is replayed again


def build_starts(loop: Loop, nights: tuple[str, ...] | None = None, runs_per_night: int = RUNS_PER_NIGHT, procs: int = 6,
                 seed: int = 20261030, agent_root: Path = REF / "agent", max_nights: int = MAX_NIGHTS) -> dict:
    """Replay `runs_per_night` runs of each night's emulator arm exactly (the M2 gym's `_replay_task`, the night's own agent) and keep
    every Act 1-2 fight they entered as a start. `nights` defaults to every finished night's arm, the newest `max_nights`
    (`nights_available`). Each night's runs are its own seeded draw (`night_seed`) and a run once replayed is kept, so a new night
    replays only its own runs. Resumable. Writes starts/starts.jsonl, its byte index, the list without documents, and meta.json last
    (with the spec `starts_stale` compares)."""
    sys.path.insert(0, str(ROOT))
    from tools.solver import coverage  # noqa: PLC0415

    nights = nights_available(max_nights=max_nights) if nights is None else tuple(nights)
    exclude = board_run_seeds()
    meta = {"nights": {}, "exclude_runs": len(exclude), "seed": seed, "runs_per_night": runs_per_night,
            "spec": start_spec(nights, runs_per_night)}
    all_runs: list[dict] = []
    sources: list[tuple[Path, set[str]]] = []
    for night in nights:
        src = LAB / "nightly" / night
        label = night.replace("-emu", "")
        try:
            summ, runs, agent_dir, engine = coverage._arm(str(src))  # noqa: SLF001
        except (OSError, ValueError, KeyError, TypeError) as error:  # skipped and recorded, so not retried at every step
            loop.log(f"starts: {night} unreadable ({type(error).__name__}); skipped")
            meta["nights"][label] = {"skipped": f"unreadable: {type(error).__name__}"}
            continue
        if engine != ENGINE:
            loop.log(f"starts: {night} played on {engine}, not {ENGINE}; skipped")
            meta["nights"][label] = {"agent": agent_dir, "skipped": f"engine {engine}"}
            continue
        all_runs += fg._eligible(runs)  # noqa: SLF001
        picks = pick_runs(runs, runs_per_night, night_seed(seed, label), exclude)
        target = loop.starts / f"replays-{label.replace('/', '-')}.jsonl"
        sources.append((target, {r["seed"] for r in picks}))
        done = {r.get("seed") or (r.get("_task") or [None, None])[1] for r in iter_jsonl(target)}  # a capped replay's row: its task
        tasks = []
        for r in picks:
            if r["seed"] in done:
                continue
            wanted = [{"id": f"{label}:{r['seed']}|f{f['floor']:02d}", "cat": f["cat"], "kind": f["kind"], "act": f["act"],
                       "floor": f["floor"], "seed": r["seed"], "ix": int(r["ix"]), "night": label} for f in fight_floors(r)]
            if wanted:
                tasks.append(("replay", r["seed"], int(r["ix"]), summ.get("config") or "a3-v1", wanted, r))
        meta["nights"][label] = {"agent": agent_dir, "runs": len(picks), "todo": len(tasks)}
        if tasks:
            loop.log(f"starts: replaying {len(tasks)} runs of {label}")
            stopped = guarded(loop, lambda t=tasks, a=agent_dir, tg=target: fg.run_tasks(
                t, fg._replay_task, tg, procs, init=fg._init_worker, init_args=(ENGINE, str(agent_root), a, str(loop.root)),  # noqa: SLF001
                mem_limit=fg.MEM_LIMIT_GB * 2**30, stop=lambda: why_blocked(loop.root, starting=False)))
            if stopped:
                return {"stopped": stopped}
    # assemble: one line per start (the document included), and a byte index so a process reads only the starts it plays; only the
    # picked runs of the nights taken (a replay file keeps every run it ever replayed)
    starts = assemble_starts(loop.starts, sources)
    meta.update({"starts": len(starts), "by_cat": dict(Counter(s["cat"] for s in starts)),
                 "by_split": dict(Counter(s["split"] for s in starts)), "fights_per_run": fights_per_run(all_runs),
                 "runs_counted": len(all_runs), "replay_same": sum(1 for s in starts if s.get("replay_same")),
                 "written": stamp()})
    write_json(loop.starts / "meta.json", meta)
    loop.log(f"starts: {len(starts)} starts {meta['by_cat']}")
    return meta


def _start_ok(s: dict) -> bool:
    """A replayed start the set keeps: it has its document and the replay's room is the kind wanted (the replay's own kind check
    knows elites and bosses only: a hallway fight is a monster room of its act); sets its split."""
    s["kind_same"] = s.get("node_type") == NODE_TYPE[s["kind"]] and int(s.get("act", 0)) == int(s["cat"][3])
    if s.get("doc") and s["kind_same"]:
        s["split"] = "val" if is_val_seed(s["seed"]) else "train"
        return True
    return False


def assemble_starts(d: Path, sources: list[tuple[Path, set[str]]], stem: str = "starts") -> list[dict]:
    """The start set from the replay files, streamed: every kept start (`_start_ok`) of the given runs of each file. starts.jsonl
    holds them file by file, each file's by id (a file's lines are held as text, never the whole set); its byte index and
    starts.lite.json (by id) are written after it. Returns the lite list. `stem` names the set: the nights' is `starts`, the teacher's
    `teacher-starts` (`rl/teacher_starts.py`)."""
    out = Path(d) / f"{stem}.jsonl"
    tmp = out.with_suffix(".tmp")
    index: dict[str, int] = {}
    lite: list[dict] = []
    with open(tmp, "wb") as fh:
        for path, seeds in sources:
            chunk: dict[str, tuple[bytes, dict]] = {}  # a run replayed twice (a line cut by a stop) counts once: its last replay
            for r in iter_jsonl(path):
                if r.get("seed") not in seeds:
                    continue
                for s in r.get("starts") or []:
                    if _start_ok(s):
                        chunk[s["id"]] = ((json.dumps(s) + "\n").encode("utf-8"), {k: v for k, v in s.items() if k != "doc"})
            for sid in sorted(chunk):
                if sid not in index:
                    index[sid] = fh.tell()
                    fh.write(chunk[sid][0])
                    lite.append(chunk[sid][1])
    os.replace(tmp, out)
    lite.sort(key=lambda s: s["id"])
    write_json(Path(d) / f"{stem}.index.json", index)
    write_json(Path(d) / f"{stem}.lite.json", lite)
    return lite


def write_start_set(d: Path, starts: list[dict]) -> None:
    """starts.jsonl (one start a line, its document included, sorted by id), its byte index and the list without documents."""
    starts = sorted(starts, key=lambda s: s["id"])
    index: dict[str, int] = {}
    out = Path(d) / "starts.jsonl"
    tmp = out.with_suffix(".tmp")
    with open(tmp, "wb") as fh:
        for s in starts:
            index[s["id"]] = fh.tell()
            fh.write((json.dumps(s) + "\n").encode("utf-8"))
    os.replace(tmp, out)
    write_json(Path(d) / "starts.index.json", index)
    write_json(Path(d) / "starts.lite.json", [{k: v for k, v in s.items() if k != "doc"} for s in starts])


_START_CACHE: dict = {}


def load_start(starts_path: str, sid: str) -> dict:
    """One start's line from the start set, through its byte index (a process never parses the whole set)."""
    cache = _START_CACHE.setdefault(starts_path, {"index": None, "docs": {}})
    if sid in cache["docs"]:
        return cache["docs"][sid]
    p = Path(starts_path)
    if cache["index"] is None:
        idx = p.with_name(p.stem + ".index.json")
        cache["index"] = read_json(idx) if idx.is_file() else None
    if cache["index"] is None:  # a small set without an index (the board's): read it whole once
        cache["docs"] = {s["id"]: s for s in fg.read_jsonl(p)}
        cache["index"] = {}
        return cache["docs"][sid]
    with open(p, "rb") as fh:
        fh.seek(cache["index"][sid])
        s = json.loads(fh.readline().decode("utf-8"))
    if len(cache["docs"]) > 256:
        cache["docs"].clear()
    cache["docs"][sid] = s
    return s


# ---------------------------------------------------------------------------------------------------------- the games (child side)

_LAST: list = [None]


def _install_capture() -> None:
    """Keep the result of the search that just returned, for the recorder: `fight_gym.play_steps` calls its grader right after the
    search, with no step in between, so the slot holds that game's result. The search itself is unchanged (the wrapper only reads)."""
    import live_search  # noqa: PLC0415

    if getattr(live_search.search_steps, "_gym_loop", False):
        return
    inner = live_search.search_steps

    def search_steps(handle, **kw):
        res = yield from inner(handle, **kw)
        _LAST[0] = res
        return res

    search_steps._gym_loop = True  # type: ignore[attr-defined]
    live_search.search_steps = search_steps


class Recorder:
    """`play_steps`'s grader: records each decision's training sample, and grades it with the exact solver where asked. It never
    changes the move: it reads the position and the search's result only."""

    def __init__(self, probe: bool, exact_wall_s: float = 0.0, keep_states: int = 0) -> None:
        import gym_lockstep  # noqa: PLC0415

        self.probe = probe
        self.grader = gym_lockstep.Grader(exact_wall_s) if exact_wall_s else None
        self.rows: list[dict] = []
        # The last `keep_states` decision positions as snapshot documents, with the teacher's root value there: when the fight
        # is lost, the oldest of them is the pre-loss start (see `write_preloss`). A position the copy refuses to export is None.
        self.states: collections.deque | None = collections.deque(maxlen=keep_states) if keep_states else None

    def __call__(self, world, k: int, best) -> dict:
        import gen_loop  # noqa: PLC0415
        import live_search  # noqa: PLC0415
        import numpy as np  # noqa: PLC0415
        import prior_client  # noqa: PLC0415

        from sts2_gym import native  # noqa: PLC0415

        res, _LAST[0] = _LAST[0], None
        if res is None or res.best is not best:
            raise RuntimeError("gym_loop: the search result of this decision was not captured (play_steps changed?)")
        probe = world.fork(live_search._PROBE_SEED)  # noqa: SLF001 -- the root prior reads this copy
        try:
            tokens = prior_client.encode_position(probe.handle, self.probe)
        finally:
            probe.close()
        visits, q = gen_loop.root_targets(res.children, MAX_ACTIONS)
        mask = native.valid_actions(world.handle, MAX_ACTIONS)
        legal = np.array([1 if mask[a] else 0 for a in range(MAX_ACTIONS)], dtype=np.int8)
        g = self.grader(world, k, best) if self.grader is not None else {"k": k}
        exact = bool(g.get("exact")) and g.get("value") is not None and "error" not in g
        sb = (g.get("solver_best") or [-1, -1])[0]
        self.rows.append({"tokens": tokens, "visits": visits, "qvals": q, "legal": legal, "sims": int(res.sims), "k": k,
                          "exact": exact, "exact_value": float(g["value"]) if exact else float("nan"),
                          "q_played": float(g["q_played"]) if exact and g.get("q_played") is not None else float("nan"),
                          "solver_best": int(sb) if exact and sb is not None else -1, "played": int(best.action)})
        if self.states is not None:
            try:
                doc = native.get_state_json(world.handle)
            except Exception as error:  # noqa: BLE001 -- ExportRefused (a step stopped on a screen) or any other: no start from here
                doc = f"{type(error).__name__}: {str(error)[:80]}"
            self.states.append((k, doc, pmix.root_value(visits, q)))
        return g


def samples_arrays(rows: list[dict]) -> dict:
    import numpy as np  # noqa: PLC0415

    return {
        "types": np.concatenate([r["tokens"].types for r in rows]).astype(np.int16),
        "ids": np.concatenate([r["tokens"].ids for r in rows]).astype(np.int32),
        "owners": np.concatenate([r["tokens"].owners for r in rows]).astype(np.int16),
        "feats": np.concatenate([r["tokens"].feats for r in rows]).astype(np.float32),
        "lengths": np.array([len(r["tokens"].types) for r in rows], dtype=np.int32),
        "visits": np.stack([r["visits"] for r in rows]).astype(np.float32),
        "qvals": np.stack([r["qvals"] for r in rows]).astype(np.float32),
        "legal": np.stack([r["legal"] for r in rows]).astype(np.int8),
        "sims": np.array([r["sims"] for r in rows], dtype=np.int32),
        "k": np.array([r["k"] for r in rows], dtype=np.int16),
        "exact": np.array([r["exact"] for r in rows], dtype=np.int8),
        "exact_value": np.array([r["exact_value"] for r in rows], dtype=np.float32),
        "q_played": np.array([r["q_played"] for r in rows], dtype=np.float32),
        "solver_best": np.array([r["solver_best"] for r in rows], dtype=np.int16),
        "played": np.array([r["played"] for r in rows], dtype=np.int16),
    }


def save_npz(path: Path, arrays: dict) -> None:
    """Atomic compressed npz: a reader never sees half a file."""
    import numpy as np  # noqa: PLC0415

    path = Path(path)
    tmp = path.with_name(path.stem + f".tmp{os.getpid()}.npz")
    np.savez_compressed(tmp, **arrays)
    os.replace(tmp, path)


def player_config(name: str):
    """The search config a player plays under: a gym arm (a3-v1, x4, x16) or a search_configs name (the judge's, once it lands)."""
    if name in fg.ARMS:
        return fg.arm_settings(name)["cfg"]
    import search_configs  # noqa: PLC0415

    return search_configs.get(name)


def fight_key(sid: str, j: int, player: str) -> str:
    return f"{player}-" + hashlib.sha256(f"{sid}|{j}".encode()).hexdigest()[:16]


def gen_game_steps(task: tuple, clock, game, ctx):
    """One generation fight, as `gym_lockstep`'s game steps: `fight_gym.play_steps` under the teacher config, every decision
    recorded; the samples go to the incoming folder as one file, the row (without the per-decision grades) to the segment's file.
    task: (arm for the deadline, start id, j, cat, info) with info {starts, incoming, player, config, exact_wall_s, record, ns}."""
    import numpy as np  # noqa: PLC0415

    arm, sid, j, cat, info = task
    start = load_start(info["starts"], sid)
    rec = None
    if info.get("record", True):
        _install_capture()
        prior = ctx.extra.get("prior_fn")
        pre = info.get("preloss") or {}
        rec = Recorder(bool(getattr(prior, "probe", False)), float(info.get("exact_wall_s") or 0.0),
                       keep_states=int(pre["back"]) + 1 if pre else 0)
    world = fg._fresh_world(start["doc"], sid, j, info["ns"])  # noqa: SLF001
    try:
        row = yield from fg.play_steps(world, start, j, player_config(info["config"]), ctx.extra, info["ns"],
                                       deadline=fg.task_deadline(task), clock=clock, progress=game.note, grader=rec)
    finally:
        world.close()
    out = {"_task": [arm, sid, j], "arm": info.get("label", arm), "config": info["config"], "start": sid, "j": j, "cat": cat,
           "player": info["player"], "split": start.get("split"), "seed": start.get("seed"), "start_hp": start.get("hp"),
           "start_potions": start.get("potions"), "start_max_hp": start.get("max_hp"), **row}
    grades = out.pop("grades", None)
    if rec is not None:
        import gym_lockstep  # noqa: PLC0415

        if rec.states is not None:
            out["preloss"] = write_preloss(info["preloss"], start, sid, j, row, rec, info["player"])
            if out["preloss"] is None:
                del out["preloss"]

        if rec.grader is not None and grades:
            out["graded"] = gym_lockstep.grade_summary(grades)
        if rec.rows:
            arrays = samples_arrays(rec.rows)
            won = bool(row["won"])
            meta = {k: out.get(k) for k in ("start", "j", "cat", "player", "config", "split", "seed", "won", "end_hp", "max_hp",
                                            "L0", "capped", "potions_used", "turns", "decisions", "start_hp", "start_potions")}
            meta.update(encounter=start.get("encounter_id"), deck=start.get("deck_size"), relics=start.get("relics"),
                        label=int(row["end_hp"]) if won else 0, value_ok=not row.get("capped"),
                        potions_held=row.get("potions_held"), source=start.get("source", "night"))
            arrays["meta"] = np.array(json.dumps(meta))
            name = fight_key(sid, j, info["player"]) + ".npz"
            save_npz(Path(info["incoming"]) / name, arrays)
            out["samples"], out["sample_file"] = len(rec.rows), name
    return out


def write_preloss(cfg: dict, start: dict, sid: str, j: int, row: dict, rec: Recorder, player: str) -> str | None:
    """The pre-loss start of a lost practice fight (memo section 5): the position `cfg["back"]` decisions before the fight's last
    decision, exported by the copy as a snapshot document, with the parent's kind, run and split and a score0 (the teacher's value
    there against the death). Written as one atomic file under `cfg["dir"]`; `pmix.merge_preloss` moves it into the set after the
    segment. Returns None for a fight that was not lost, else a word the segment counts: ok; short (fewer decisions than needed);
    refused (the copy would not export that position: no start is made from it); chained (the parent is itself a pre-loss start)."""
    if row.get("won") or row.get("capped"):
        return None
    if start.get("source") == "preloss":
        return "chained"
    if len(rec.rows) < int(cfg["min_decisions"]) or len(rec.states) < int(cfg["back"]) + 1:
        return "short"
    k, doc, root = rec.states[0]
    if not isinstance(doc, dict):
        return "refused"
    out = pmix.preloss_start(start, sid, j, k, doc, root, player, stamp())
    path = Path(cfg["dir"]) / (fight_key(sid, j, player) + ".json")
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + f".tmp{os.getpid()}")
    tmp.write_text(json.dumps(out, separators=(",", ":")), encoding="utf-8")
    os.replace(tmp, path)
    return "ok"


# ---------------------------------------------------------------------------------------------------------- the buffer


def compact(loop: Loop, segment: int, value_labels: bool = True) -> dict | None:
    """Merge the incoming fight files into one shard (`buffer/shards/sNNNNNN.npz`) and index it; the fight files go. Run after a
    segment and at start-up (a killed segment's finished fights are kept).

    `value_labels` False (a segment stopped before its end, or killed): its fights keep their policy targets but lose their value
    labels. The fights still running at a stop are lost, and the long ones are the likelier to be running, so the fights a cut
    segment did finish lean towards short ones (quick deaths); their outcomes would bias the value head. The visit targets do not
    depend on how the fight ended."""
    import numpy as np  # noqa: PLC0415

    files = sorted(loop.incoming.glob("*.npz"))
    files = [f for f in files if ".tmp" not in f.name]
    for f in loop.incoming.glob("*.tmp*"):
        f.unlink(missing_ok=True)
    if not files:
        return None
    parts, metas, rows_of = [], [], []
    for f in files:
        try:
            with np.load(f) as z:
                a = {k: z[k] for k in z.files}
        except (OSError, ValueError, EOFError):
            loop.log(f"buffer: {f.name} unreadable; dropped")
            f.unlink(missing_ok=True)
            continue
        metas.append(json.loads(str(a.pop("meta"))))
        rows_of.append(int(a["lengths"].size))
        parts.append(a)
    if not parts:
        return None
    merged = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
    n = sum(rows_of)
    merged["fight"] = np.repeat(np.arange(len(parts), dtype=np.int32), rows_of)
    merged["label"] = np.repeat(np.array([m["label"] for m in metas], dtype=np.int16), rows_of)
    merged["value_ok"] = np.repeat(np.array([bool(m["value_ok"]) and value_labels for m in metas], dtype=np.int8), rows_of)
    merged["val"] = np.repeat(np.array([m.get("split") == "val" for m in metas], dtype=np.int8), rows_of)
    merged["fights_meta"] = np.array(json.dumps(metas))
    name = f"s{segment:06d}-{int(time.time())}.npz"
    save_npz(loop.shards / name, merged)
    entry = {"shard": name, "segment": segment, "rows": n, "fights": len(parts), "bytes": (loop.shards / name).stat().st_size,
             "players": dict(Counter(m["player"] for m in metas)), "cats": dict(Counter(m["cat"] for m in metas)),
             "val_rows": int(merged["val"].sum()), "value_rows": int(merged["value_ok"].sum()), "t": stamp()}
    with open(loop.root / "buffer" / "index.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(entry) + "\n")
    for f in files:
        f.unlink(missing_ok=True)
    return entry


def retain(loop: Loop, max_rows: int, max_days: float, max_gb: float) -> dict:
    """The buffer's retention window and disk cap: the newest shards are kept while their rows stay within `max_rows`, their age
    within `max_days` and their bytes within `max_gb`; older shards are deleted (the oldest first)."""
    shards = sorted(loop.shards.glob("s*.npz"), key=lambda p: p.name, reverse=True)
    index = {e["shard"]: e for e in fg.read_jsonl(loop.root / "buffer" / "index.jsonl")}
    rows = size = 0
    gone = []
    now = time.time()
    for p in shards:
        if ".tmp" in p.name:
            continue
        e = index.get(p.name, {})
        r, b = int(e.get("rows", 0)), p.stat().st_size
        age_d = (now - p.stat().st_mtime) / 86400
        if rows + r > max_rows or size + b > max_gb * 2**30 or age_d > max_days:
            p.unlink(missing_ok=True)
            gone.append(p.name)
            continue
        rows, size = rows + r, size + b
    if gone:
        loop.log(f"buffer: retention removed {len(gone)} shards; kept {rows} rows, {size / 2**30:.2f} GB")
    return {"rows": rows, "gb": round(size / 2**30, 3), "removed": len(gone)}


def prune_checkpoints(loop: Loop, keep: int = 48) -> None:
    """The newest `keep` checkpoints stay (a promoted one is copied to players/ first, so it never goes with them)."""
    done = sorted(loop.checkpoints.glob("c[0-9][0-9][0-9][0-9]"))
    for p in done[:-keep] if keep > 0 else []:
        shutil.rmtree(p, ignore_errors=True)


# ---------------------------------------------------------------------------------------------------------- the weights


def weights_stale(loop: Loop) -> str | None:
    """Why the kind weights must be computed again (None: they are current): none yet, a start set written since, or another
    incumbent (the loss is the incumbent's)."""
    w = read_json(loop.starts / "weights.json", None)
    if not w:
        return "no weights"
    meta = read_json(loop.starts / "meta.json", {}) or {}
    if w.get("starts_written") != meta.get("written"):
        return "start set rebuilt"
    if w.get("player") != loop.state["incumbent"]:
        return f"incumbent {w.get('player')} to {loop.state['incumbent']}"
    return None


def kind_weights(loop: Loop, procs: int = 6, per_kind: int = 50, seeds: int = 2, k: int = 16, prior: str = "cpu") -> dict:
    """The start mix (gym-build P2): per kind, the fights a run enters times the mean L0 a fight loses (a death loses all of it),
    the loss measured on `per_kind` training starts x `seeds` games at a3-v1 with the incumbent. The incumbent's games are kept
    across start sets: a kind is topped up to `per_kind` x `seeds` games from its starts in a fixed order, so a new night or window
    plays nothing and a new incumbent plays its own. Resumable; writes weights.json."""
    meta = read_json(loop.starts / "meta.json", {})
    lite = read_json(loop.starts / "starts.lite.json", [])
    by: dict[str, list[str]] = defaultdict(list)
    for s in lite:
        if s["split"] == "train":
            by[s["cat"]].append(s["id"])
    target = loop.starts / "weights_games.jsonl"
    inc, inc_dir = loop.incumbent()
    mine = [r for r in fg.read_jsonl(target) if r.get("player") == inc and r.get("config") == "a3-v1"]
    played = {(r["start"], int(r["j"])) for r in mine}
    have = Counter(r["cat"] for r in mine)
    info = {"starts": str(loop.starts / "starts.jsonl"), "incoming": str(loop.incoming), "player": inc, "config": "a3-v1",
            "record": False, "ns": NS_WEIGHTS}
    tasks = []
    for c, ids in sorted(by.items()):
        need = per_kind * seeds - have[c]
        order = sorted(ids)
        random.Random(f"weights:{c}").shuffle(order)
        for sid in order:
            for j in range(seeds):
                if need > 0 and (sid, j) not in played:
                    tasks.append(("a3-v1", sid, j, c, info))
                    need -= 1
            if need <= 0:
                break
    if tasks:
        loop.log(f"weights: {inc} plays {len(tasks)} games " + str(dict(Counter(t[3] for t in tasks))))
        stopped = play_tasks(loop, tasks, target, inc_dir, procs, k, prior, "weights")
        if stopped:
            return {"stopped": stopped}
    rows = [r for r in fg.read_jsonl(target) if r.get("player") == inc and r.get("config") == "a3-v1"]
    loss: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        start_l0 = fg.l0(True, int(r.get("start_hp") or 0), int(r.get("start_potions") or 0), int(r.get("start_max_hp") or 1))
        loss[r["cat"]].append(start_l0 - float(r.get("L0", 0.0)))
    fpr = meta.get("fights_per_run") or {}
    raw = {c: fpr.get(c, 0.0) * (sum(v) / len(v)) for c, v in loss.items() if v}
    tot = sum(raw.values()) or 1.0
    w = {"weights": {c: raw[c] / tot for c in sorted(raw)}, "fights_per_run": fpr,
         "mean_l0_lost": {c: sum(v) / len(v) for c, v in sorted(loss.items())}, "games": {c: len(v) for c, v in sorted(loss.items())},
         "win": {c: sum(1 for r in rows if r["cat"] == c and r.get("won")) / max(1, len(loss[c])) for c in sorted(loss)},
         "player": inc, "config": "a3-v1", "starts_written": meta.get("written"), "written": stamp()}
    write_json(loop.starts / "weights.json", w)
    loop.log("weights: " + ", ".join(f"{c} {x:.3f}" for c, x in w["weights"].items()))
    return w


# ---------------------------------------------------------------------------------------------------------- playing


def guarded(loop: Loop, fn):
    """Run one heavy step with the daemon's QUIET.lock line held (refreshed by the step's stop polls), then drop it."""
    hold_quiet(True)
    try:
        return fn()
    finally:
        hold_quiet(False)


def _champion(player_dir: Path):
    import combat_prior  # noqa: PLC0415

    return combat_prior.find(player_dir)


class Server:
    """The prior server for one player (P1's, on the card), started only while GPU_OK allows it; None means the CPU prior."""

    def __init__(self, loop: Loop, player_dir: Path, name: str) -> None:
        self.loop, self.handle, self.opts = loop, None, None
        if os.environ.get("GYM_LOOP_NO_GPU"):
            return
        if not take_gpu(TAG):
            holder = (_lines(GPU_OK) or ["?"])[0].split()[0]  # the first GPU_OK line holds the card (a gate, the night)
            loop.log(f"gpu: {name}: the card is held by {holder}; the CPU prior plays")
            if loop.clock is not None:
                loop.clock.note(f"gpu-denied:{holder}")
            return
        from prior_launch import start_prior_server  # noqa: PLC0415

        champ = _champion(player_dir)
        self.handle = start_prior_server(GPU_PYTHON, champ.policy, lib_path=ROOT / "out", kill_file=GPU_OFF,
                                         stats_path=loop.games / f"server-{name}.json", priority="below")
        if self.handle is None:
            drop_gpu(TAG)
            loop.log("server: did not start; the CPU prior plays")
            return
        self.opts = {"endpoint": self.handle.endpoint, "authkey": self.handle.authkey, "policy_sha256": champ.policy_sha256}

    @property
    def pid(self) -> int | None:
        return self.handle.proc.pid if self.handle is not None and self.handle.proc is not None else None

    def stop(self) -> dict:
        stats = {}
        if self.handle is not None:
            stats = self.handle.read_stats()
            self.handle.stop()
            self.handle = None
            drop_gpu(TAG)
        return stats


def play_tasks(loop: Loop, tasks: list, target: Path, player_dir: Path, procs: int, k: int, prior: str, what: str,
               steps: str = "gym_loop:gen_game_steps", daemon=None, extra_opts: dict | None = None) -> str | None:
    """Play `tasks` in P1's lockstep processes (prior on the card when allowed), rows appended to `target`, under the QUIET line.
    Returns why it stopped, or None when every task has its row."""
    import gym_lockstep  # noqa: PLC0415

    init_args = (ENGINE, str(player_dir.parent), player_dir.name, str(loop.root))
    server = Server(loop, player_dir, what) if prior == "server" else None
    opts = {"prior": "server" if server is not None and server.opts else "cpu", "server": server.opts if server else None,
            "game_steps": steps, "worlds": "reload", "grade_wall_s": 0, "paths": [str(ROOT / "rl")], **(extra_opts or {})}
    judge_server = None
    if opts.get("judge"):  # the judge's leaves: on its own server on the card beside the prior's, as the judge's arm plays them
        j = {"path": str(opts["judge"]["path"]), "sha256": sha256_file(Path(opts["judge"]["path"]))}
        if server is not None and server.opts:
            from prior_launch import start_prior_server  # noqa: PLC0415

            judge_server = start_prior_server(GPU_PYTHON, j["path"], lib_path=ROOT / "out", kill_file=GPU_OFF,
                                              stats_path=loop.games / f"judge-server-{what}.json", priority="below")
            if judge_server is not None:
                j.update(endpoint=judge_server.endpoint, authkey=judge_server.authkey)
            else:
                loop.log(f"{what}: the judge's server did not start; the judge plays on the CPU")
        opts["judge"] = j
    if daemon is not None:
        daemon.server_pid = server.pid if server else None
        daemon.judge_pid = judge_server.proc.pid if judge_server is not None and judge_server.proc is not None else None
    t0 = time.perf_counter()
    last = [time.time()]

    def stop() -> str | None:
        if time.time() - last[0] > 1800:
            hold_quiet(True)  # refresh the line every half hour while the step runs
            last[0] = time.time()
        return why_blocked(loop.root, starting=False)

    try:
        hold_quiet(True)
        stopped, stats = gym_lockstep.run_lockstep(tasks, target, procs, min(k, gym_lockstep.K_MAX), init_args, opts, stop=stop,
                                                   log=lambda m: loop.log(f"{what}: {m}"))
    finally:
        hold_quiet(False)
        if judge_server is not None:
            judge_server.stop()
        sstats = server.stop() if server is not None else {}
        if daemon is not None:
            daemon.server_pid = daemon.judge_pid = None
    wall = time.perf_counter() - t0
    with open(loop.games / "steps.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps({"t": stamp(), "what": what, "tasks": len(tasks), "wall_s": round(wall, 1), "stopped": stopped,
                             "prior": opts["prior"], "procs": procs, "k": k, "server": {x: sstats.get(x) for x in
                             ("replays", "rows", "replay_ms_mean", "busy_s") if x in sstats},
                             "processes": [{x: s.get(x) for x in ("games", "rounds", "requests", "wall_s", "net_s", "games_s",
                                                                  "cpu_s")} for s in stats]}) + "\n")
    return stopped


# ---------------------------------------------------------------------------------------------------------- generation


def sample_tasks(loop: Loop, player: str, budget_s: float, teacher: str, exact_wall_s: float, rnd: random.Random,
                 preloss: bool = True) -> list:
    """A segment's fights, until the estimated own seconds reach `budget_s`: a kind by the start mix's weights (`weights.json`),
    a source by the kind's human share, a start by the regret rule (`practice_mix`: the rank of the start's score among its kind
    and side, with a staleness term so starts not drawn lately come up), game j the start's next unplayed one. Starts of the exam
    (`exam/exclude_runs.json`) are never drawn. Long fights first, so the segment's tail is the short ones. What was drawn is left
    in `loop.mix` for the log; `preloss`: the games keep the position three decisions before a loss (`write_preloss`)."""
    weights = (read_json(loop.starts / "weights.json", {}) or {}).get("weights") or {}
    cfg = pmix.config(loop.starts)
    ex = pmix.read_exclusion(loop.root)
    pools, dropped = pmix.load_pools(loop.root, ex)
    plays = pmix.sync_plays(loop.root, loop.log)
    human = pmix.read_human_values(loop.starts) if cfg["regret"]["mode"] in pmix.HUMAN_MODES else None
    score = pmix.scores(plays, int(cfg["regret"]["n_last"]), cfg["regret"]["mode"], human)
    seg = int(loop.state["segment"])
    last_seg = loop.state["last_seg"]
    known = {s["id"] for items in pools.values() for s in items}
    if len(last_seg) > 2 * len(known) + 1000:  # starts no longer in any pool leave the staleness record
        for sid in [x for x in last_seg if x not in known]:
            del last_seg[sid]
    sampler = pmix.Sampler(pools, weights, cfg, score, last_seg, seg)
    drawn: list[dict] = []
    spent = 0.0
    while spent < budget_s and len(drawn) < 5000:
        d = sampler.draw(rnd)
        drawn.append(d)
        spent += est_seconds(loop, d["cat"])
    for d in drawn:
        last_seg[d["id"]] = seg
    base = {"incoming": str(loop.incoming), "player": player, "config": teacher, "exact_wall_s": exact_wall_s, "record": True,
            "ns": NS_GEN}
    if preloss:
        base["preloss"] = {"dir": str(loop.starts / "preloss-new"), "back": int(cfg["preloss"]["back"]),
                           "min_decisions": int(cfg["preloss"]["min_decisions"])}
    infos: dict[str, dict] = {}
    nj = loop.state["next_j"]
    tasks = []
    for d in drawn:
        info = infos.setdefault(d["file"], {**base, "starts": d["file"]})
        j = int(nj.get(d["id"], 0))
        nj[d["id"]] = j + 1
        tasks.append((teacher, d["id"], j, d["cat"], info))
    tasks.sort(key=lambda t: -est_seconds(loop, t[3]))
    loop.mix = {**pmix.mix_summary(drawn), "left_out": dict(dropped), "excluded_ids": len(ex), "scored_starts": len(score)}
    return tasks


def est_seconds(loop: Loop, cat: str) -> float:
    s, n = loop.state["kind_s"].get(cat, [0.0, 0])
    return s / n if n >= 5 else MEAN_X16_S.get(cat, 30.0)


def learn_seconds(loop: Loop, rows: list[dict]) -> None:
    for r in rows:
        if r.get("cat") and r.get("seconds") is not None and not r.get("capped"):
            s, n = loop.state["kind_s"].get(r["cat"], [0.0, 0])
            loop.state["kind_s"][r["cat"]] = [s + float(r["seconds"]), n + 1]


# ---------------------------------------------------------------------------------------------------------- the gate


def mcnemar_one_sided(b: int, c: int) -> float:
    """P(at least b candidate-only wins of the b + c discordant pairs | no difference): the exact one-sided McNemar test."""
    n = b + c
    if n == 0:
        return 1.0
    return min(1.0, sum(math.comb(n, i) for i in range(b, n + 1)) / 2**n)


def stage1_pairs(rule: dict) -> int:
    """How many stage-1 pairs of seed indices the rule's stage 2 covers (five: 10 seed indices, two to a pair)."""
    return max(1, rule["seeds"] // rule["stage1_seeds"])


def stage1_js(rule: dict, pair: int) -> tuple[int, ...]:
    """The seed indices of stage 1's pair number `pair` (wrapped): pair 0 is (0, 1), pair 1 is (2, 3), ..."""
    w = rule["stage1_seeds"]
    p = int(pair) % stage1_pairs(rule)
    return tuple(range(p * w, p * w + w))


def next_gate_pair(rule: dict, pair: int) -> int:
    """The pair after a promotion: the next one, wrapping. The promoted checkpoint was picked on `pair`; the next candidate
    meets it on a pair it was not picked on."""
    return (int(pair) + 1) % stage1_pairs(rule)


def paired(cand: list[dict], inc: list[dict], seeds: int, js: tuple[int, ...] | None = None) -> dict:
    """The pairs (start, j) both arms played, j < seeds or, given `js`, j in js: wins (a capped game is lost), discordant counts,
    L0 difference."""
    want = (lambda j: j in js) if js is not None else (lambda j: j < seeds)  # noqa: E731
    a = {(r["start"], int(r["j"])): r for r in cand if want(int(r["j"]))}
    b = {(r["start"], int(r["j"])): r for r in inc if want(int(r["j"]))}
    keys = sorted(set(a) & set(b))
    wa = [bool(a[x].get("won")) and not a[x].get("capped") for x in keys]
    wb = [bool(b[x].get("won")) and not b[x].get("capped") for x in keys]
    la = [float(a[x].get("L0", 0.0)) for x in keys]
    lb = [float(b[x].get("L0", 0.0)) for x in keys]
    n = len(keys)
    d_win = [float(x) - float(y) for x, y in zip(wa, wb)]
    d_l0 = [x - y for x, y in zip(la, lb)]
    mw, sw, _ = fg.mean_se(d_win)
    ml, sl, _ = fg.mean_se(d_l0)
    return {"pairs": n, "cand_wins": sum(wa), "inc_wins": sum(wb), "cand_only": sum(1 for x, y in zip(wa, wb) if x and not y),
            "inc_only": sum(1 for x, y in zip(wa, wb) if y and not x), "win_diff": mw, "win_se": sw, "l0_diff": ml, "l0_se": sl,
            "cand_rate": sum(wa) / n if n else None, "inc_rate": sum(wb) / n if n else None,
            "capped": sum(1 for x in keys if a[x].get("capped")) + sum(1 for x in keys if b[x].get("capped"))}


def decide(cand: list[dict], inc: list[dict], stage: int, rule: dict | None = None, pair: int = 0,
           pool: list[tuple[str, dict]] | None = None) -> dict:
    """The fixed rule on the games so far. Stage 1 (on stage-1 pair number `pair`): 'continue' or 'reject'. Stage 2: 'promote' or
    'reject'. A rule with a `screen` is v3 (`rl/gym_gate.py`); `pool` is then the earlier stage-2 candidates of this incumbent as
    [(name, {(start, j): candidate win - incumbent win})], oldest first, this candidate not among them. A v2 rule (`futility_points`) is
    decided as it always was."""
    rule = rule or PROMOTION_RULE
    v3 = "screen" in rule
    if stage == 1:
        js = stage1_js(rule, pair)
        p = paired(cand, inc, rule["stage1_seeds"], js)
        need = rule["starts"] * len(js)
        if p["pairs"] < need:
            return {"verdict": "incomplete", "stage": 1, **p}
        if v3:
            sc = gg.screen_stage1(gg.pair_diffs(cand, inc, lambda j: j in js), rule["screen"])
            bad = bool(sc["reject"])
            why = (f"stage 1: the 95% {rule['screen'].get('bound', 'upper')} bound {100 * sc['bound']:+.1f} points is under the "
                   f"{rule['screen']['points']:+.1f} that marks a candidate clearly worse") if bad else None
            return {"verdict": "reject" if bad else "continue", "stage": 1, "why": why, "diff_se": sc["se"], "diff_bound": sc["bound"], **p}
        go = p["win_diff"] is not None and 100 * p["win_diff"] >= rule["futility_points"]
        return {"verdict": "continue" if go else "reject", "stage": 1, "why": None if go else
                f"stage 1: {100 * (p['win_diff'] or 0):+.1f} points, under the {rule['futility_points']:+.1f} needed to go on", **p}
    p = paired(cand, inc, rule["seeds"])
    if p["pairs"] < rule["starts"] * rule["seeds"]:
        return {"verdict": "incomplete", "stage": 2, **p}
    pval = mcnemar_one_sided(p["cand_only"], p["inc_only"])
    if v3:
        r = gg.stage2(gg.pair_diffs(cand, inc, lambda j: j < rule["seeds"]), [d for _name, d in pool or []], pval, p["l0_diff"], rule)
        pl = r["pool"]
        keep = int(rule["pool"]["k"]) - 1
        return {"verdict": "promote" if r["ok"] else "reject", "stage": 2, "p": pval, "why": r["why"], "path": r["path"],
                "diff_se": r["diff_se"], "diff_lb": r["diff_lb"], "pool_k": pl["k"], "pool_diff": pl["diff"], "pool_se": pl["se"],
                "pool_p": pl["p"], "pool_prior": [name for name, _d in (pool or [])][-keep:] if keep > 0 else [], **p}
    ok_l0 = (p["l0_diff"] or 0.0) >= 0.0 or not rule["l0_not_worse"]
    ok = pval < rule["alpha"] and ok_l0
    why = None if ok else ("McNemar p %.3f >= %.2f" % (pval, rule["alpha"]) if pval >= rule["alpha"] else "mean L0 lower")
    return {"verdict": "promote" if ok else "reject", "stage": 2, "p": pval, "why": why, **p}


def pool_members(loop: "Loop", inc: str, ctag: str, rule: dict, cand: str) -> list[tuple[str, dict]]:
    """The rule's pool before `cand`: the newest `pool.k - 1` stage-2 candidates of the ledger (`gate/verdicts.jsonl`) with this
    incumbent and this rule's version, each as its pair differences against the incumbent on its own engine's games ([(name, diffs)],
    oldest first). A member whose games are gone is left out. A promotion changes the incumbent, so the pool restarts there."""
    keep = int(rule["pool"]["k"]) - 1
    if keep <= 0:
        return []
    rows = [r for r in gg.read_ledger(loop.gates / "verdicts.jsonl")
            if r.get("stage") == 2 and r.get("verdict") in ("reject", "promote") and r.get("incumbent") == inc
            and r.get("rule") == rule["name"] and r.get("candidate") != cand]
    out: list[tuple[str, dict]] = []
    seen: set[str] = set()
    for r in reversed(rows):
        if r["candidate"] in seen:
            continue
        seen.add(r["candidate"])
        cf = loop.gates / f"{r['candidate']}-{ctag}-{r['engine']}.jsonl"
        nf = loop.gates / f"inc-{inc}-{ctag}-{r['engine']}.jsonl"
        if not (cf.is_file() and nf.is_file()):
            continue
        d = gg.pair_diffs(fg.read_jsonl(cf), fg.read_jsonl(nf), lambda j: j < rule["seeds"])
        if len(d) < rule["starts"] * rule["seeds"]:
            continue
        out.append((r["candidate"], d))
        if len(out) >= keep:
            break
    return out[::-1]


def promotion_guard(loop: "Loop", v: dict, rule: dict | None = None, now: dt.datetime | None = None) -> dict:
    """At most one promotion in `guard_h` hours: a promote inside the window becomes a reject that says why (`guard`, and the path it
    passed by), the games and the pool untouched. A rule with no `guard_h`, no promotion yet or an unreadable time: no guard."""
    rule = rule or PROMOTION_RULE
    hrs = float(rule.get("guard_h") or 0.0)
    if v.get("verdict") != "promote" or hrs <= 0 or not loop.state.get("promotions"):
        return v
    last_p = loop.state["promotions"][-1]
    last = gg.parse_stamp(last_p.get("t"))
    age = None if last is None else ((now or utc()) - last).total_seconds() / 3600.0
    if age is None or age >= hrs:
        return v
    return {**v, "verdict": "reject", "guard": f"{last_p.get('version')} was promoted {age:.1f} h ago (one promotion in {hrs:g} h)",
            "why": f"guard: {last_p.get('version')} was promoted {age:.1f} h ago, one promotion in {hrs:g} h; the rule had passed by path {v.get('path')}",
            "guarded_path": v.get("path"), "path": None}


def board_starts() -> list[dict]:
    return [s for s in fg.read_jsonl(BOARD_DIR / "starts.jsonl") if s["cat"] == PROMOTION_RULE["cat"]][: PROMOTION_RULE["starts"]]


def gate_tasks(version: str, config: str, seeds: range, done: set, label: str) -> list:
    info = {"starts": str(BOARD_DIR / "starts.jsonl"), "incoming": "", "player": version, "config": config, "record": False,
            "ns": PROMOTION_RULE["ns"], "label": label}
    return [("a3-v1", s["id"], j, s["cat"], info) for j in seeds for s in board_starts() if (s["id"], j) not in done]


def engine_tag() -> str:
    return sha256_file(ROOT / "out" / "Sts2Emulator.dll")[:12]


def reuse_incumbent_games(loop: Loop, inc: str, inc_file: Path, js: tuple[int, ...], ctag: str, eng: str) -> int:
    """The incumbent's stage-1 reference: its games on seed indices `js` are taken from the gate file of the checkpoint it was
    promoted from (the same weights, config, judge and engine; the file name says so) into its own kept games, for the pairs not
    already there. Returns how many rows were added. A version with no such file (version 0, another engine or config) gets none,
    and the gate plays the missing games itself, once."""
    ckpt = next((x["checkpoint"] for x in loop.state["promotions"] if x["version"] == inc), None)
    src = loop.gates / f"{ckpt}-{ctag}-{eng}.jsonl" if ckpt else None
    if src is None or not src.is_file():
        return 0
    have = {(r["start"], int(r["j"])) for r in fg.read_jsonl(inc_file)}
    take = [{**r, "arm": inc, "player": inc, "reused_from": ckpt} for r in fg.read_jsonl(src)
            if int(r["j"]) in js and (r["start"], int(r["j"])) not in have]
    if take:
        end = inc_file.read_bytes()[-1:] if inc_file.is_file() else b"\n"
        with open(inc_file, "a", encoding="utf-8", newline="\n") as fh:
            fh.write("" if end in (b"\n", b"") else "\n")  # a line cut by a stop must not swallow the first copied row
            fh.writelines(json.dumps(r) + "\n" for r in take)
    return len(take)


def _rule_note(v: dict) -> str:
    """What v3 adds to a gate line: stage 1's upper bound, stage 2's lower bound, the pool and the path ("" for a v2 verdict)."""
    if v.get("verdict") == "incomplete":
        return ""
    if v.get("stage") == 1 and v.get("diff_bound") is not None:
        return f"; 95% upper bound {100 * v['diff_bound']:+.1f} points"
    if v.get("stage") == 2 and "pool_k" in v:
        pd, pp = v.get("pool_diff"), v.get("pool_p")
        pool = ("pool of %d: %s (SE %.2f), p %s" % (v["pool_k"], "n/a" if pd is None else "%+.2f points" % (100 * pd),
                                                   100 * (v.get("pool_se") or 0.0), "n/a" if pp is None else "%.3f" % pp))
        tail = "promote by path %s" % v["path"] if v.get("path") else "reject: %s" % v.get("why")
        return "; lower bound %+.2f points; %s; %s" % (100 * (v.get("diff_lb") or 0.0), pool, tail)
    return ""


def run_gate(loop: Loop, cand: str, cand_dir: Path, config: str, procs: int, k: int, prior: str, daemon=None,
             judge: str | None = None) -> dict:
    """Score a checkpoint against the incumbent with the fixed rule (resumable: the games on disk are kept)."""
    inc, inc_dir = loop.incumbent()
    eng = engine_tag()
    # the player is a parameter: its search config, and the judge checkpoint a judge config's leaves read (passed to the
    # lockstep processes as gym_lockstep's `judge` option, on the CPU); the incumbent's kept games are per config and judge
    ctag = config + (f"+j{sha256_file(Path(judge))[:8]}" if judge else "")
    xo = {"judge": {"path": str(judge)}} if judge else None
    inc_file = loop.gates / f"inc-{inc}-{ctag}-{eng}.jsonl"
    cand_file = loop.gates / f"{cand}-{ctag}-{eng}.jsonl"
    rule = PROMOTION_RULE
    pair = loop.gate_pair()
    js = stage1_js(rule, pair)
    out: dict = {"candidate": cand, "incumbent": inc, "config": config, "engine": eng, "rule": rule["name"], "stage1_pair": pair,
                 "stage1_js": list(js)}
    for stage, seeds in ((1, js), (2, range(rule["seeds"]))):
        if stage == 1:
            n = reuse_incumbent_games(loop, inc, inc_file, js, ctag, eng)
            if n:
                loop.log(f"gate {cand}: stage 1, {n} of {inc}'s games on j {list(js)} taken from its promotion gate's games")
        for who, d, f in ((inc, inc_dir, inc_file), (cand, cand_dir, cand_file)):
            done = {(r["start"], int(r["j"])) for r in fg.read_jsonl(f)}
            tasks = gate_tasks(who, config, seeds, done, label=who)
            if tasks:
                loop.log(f"gate {cand}: stage {stage}, {who} plays {len(tasks)} games")
                stopped = play_tasks(loop, tasks, f, d, procs, k, prior, f"gate-{who}", daemon=daemon, extra_opts=xo)
                if stopped:
                    return {**out, "verdict": "stopped", "why": stopped}
        pool = pool_members(loop, inc, ctag, rule, cand) if stage == 2 and "pool" in rule else None
        v = decide(fg.read_jsonl(cand_file), fg.read_jsonl(inc_file), stage, pair=pair, pool=pool)
        out.update(v)
        loop.log(f"gate {cand}: stage {stage}{f' (j {list(js)})' if stage == 1 else ''} {v['verdict']}: cand {v.get('cand_wins')} "
                 f"inc {v.get('inc_wins')} of {v.get('pairs')}"
                 f" ({100 * (v.get('win_diff') or 0):+.1f} points, SE {100 * (v.get('win_se') or 0):.1f})"
                 + (f" p {v['p']:.3f}" if "p" in v else "") + _rule_note(v))
        if v["verdict"] != "continue":
            break
    out["cand_file"], out["inc_file"] = cand_file.name, inc_file.name
    return out


# ---------------------------------------------------------------------------------------------------------- the gate under rule v4


def v4_tasks(fe, starts_file: Path, lite: dict, who: str, cfg: str, done: set, draws: int, js: tuple[int, ...] | None = None) -> list:
    """The exam games of `who` still to play on `lite`'s starts (documents read from `starts_file`), draws 0..draws-1 (or `js`), long
    fights first, as `fight_exam.exam_tasks` orders them."""
    info = {"starts": str(starts_file), "player": who, "config": cfg, "ns": fe.NS, "label": who}
    rank = {"boss": 0, "elite": 1, "hallway": 2}
    tasks = [("a3-v1", s["id"], j, s["cat"], info) for s in lite.values() for j in (range(draws) if js is None else js)
             if (s["id"], j) not in done]
    tasks.sort(key=lambda t: (rank.get(lite[t[1]].get("kind"), 0), t[2], t[1]))
    return tasks


def exam_rows(path: Path, starts: dict, cfg: str) -> list[dict]:
    """The rows of an exam games file that are games of `starts` under `cfg`."""
    return [r for r in fg.read_jsonl(path) if r.get("start") in starts and r.get("config", cfg) == cfg]


def usable_index(rows: list[dict], eng: str, ok: set, want=None) -> dict:
    """{(start, j): row} of the rows played on this engine or on one whose identity check passed (`ok`); a row on this engine wins."""
    old = gg.indexed([r for r in rows if r.get("engine") in ok and r.get("engine") != eng], want)
    cur = gg.indexed([r for r in rows if r.get("engine") == eng], want)
    return {**old, **cur}


def identity_ledger(exam: Path) -> list[dict]:
    return fg.read_jsonl(exam / "identity.jsonl")


def ensure_identity(loop: Loop, who: str, who_dir: Path, games: Path, starts_file: Path, lite: dict, eng: str, rule: dict, procs: int, k: int,
                    daemon=None, tag: str = "exam", sub: str = "", folder: Path | None = None) -> set | str:
    """The engines whose stored rows of `who` in `games` can be used on this engine `eng`: `eng` itself, and every other engine whose
    check has passed. A new engine gets its check once per player (`identity.n` stored games replayed by `who` under the exam's config
    and compared on outcome, end HP and move hash: `gym_gate.identity_check`), recorded in `exam/identity.jsonl` and the replayed rows kept
    in `exam/identity-<who>-<old>-<eng>.jsonl`; a check that fails leaves that engine's rows unused (the games are played again on this
    engine). Returns the set, or "stopped" when a stop cut the replay short. `folder`: where the ledger and the replays live (rule v5's
    `gate5/`; the exam's folder by default)."""
    exam = Path(folder) if folder is not None else loop.root / "exam"
    cfg = rule["config"]
    rows = exam_rows(games, lite, cfg)
    ok = {eng}
    for old in sorted({r.get("engine") for r in rows if r.get("engine")} - {eng}):
        led = [r for r in identity_ledger(exam) if (r.get("player"), r.get("from_engine"), r.get("to_engine"), r.get("set", "")) == (who, old, eng, sub)]
        if led:
            if led[-1]["ok"]:
                ok.add(old)
            continue
        stored = gg.indexed([r for r in rows if r.get("engine") == old])
        keys = gg.identity_sample(stored, int(rule["identity"]["n"]), f"{rule['identity']['seed']}:{who}:{old}:{sub}")
        target = exam / f"identity-{who}-{old}-{eng}{('-' + sub) if sub else ''}.jsonl"
        have = {(r["start"], int(r["j"])) for r in fg.read_jsonl(target)}
        everything = {(s, j) for s in lite for j in range(1 + max(j for _s, j in stored))}
        tasks = v4_tasks(_fight_exam(), starts_file, lite, who, cfg, everything - set(keys) | have, 1 + max(j for _s, j in stored))
        if tasks:
            loop.log(f"gate {tag}: identity check of {who}'s rows from engine {old} on {eng}: {len(tasks)} of {len(keys)} games replayed")
            stopped = play_tasks(loop, tasks, target, who_dir, procs, k, rule["prior"], f"identity-{who}", steps="fight_exam:exam_game_steps",
                                 daemon=daemon)
            if stopped:
                return "stopped"
        res = gg.identity_check({x: stored[x] for x in keys}, gg.indexed(fg.read_jsonl(target)))
        with open(exam / "identity.jsonl", "a", encoding="utf-8", newline="\n") as fh:
            fh.write(json.dumps({"t": stamp(), "player": who, "from_engine": old, "to_engine": eng, "set": sub, **res}) + "\n")
        loop.log(f"gate {tag}: identity check of {who}'s rows from engine {old} on {eng}: {res['same']} of {res['n']} games identical"
                 + ("" if res["ok"] else "; NOT identical, those rows are not used and the games are played again"))
        if res["ok"]:
            ok.add(old)
    return ok


def exam_play(loop: Loop, who: str, who_dir: Path, games: Path, starts_file: Path, lite: dict, eng: str, ok: set, draws: int, rule: dict,
              procs: int, k: int, daemon=None, js: tuple[int, ...] | None = None) -> str | None:
    """Play the exam games `who` has no usable row of (draws 0..draws-1, or `js`) into `games`; the stop reason or None."""
    cfg = rule["config"]
    done = set(usable_index(exam_rows(games, lite, cfg), eng, ok))
    tasks = v4_tasks(_fight_exam(), starts_file, lite, who, cfg, done, draws, js)
    if not tasks:
        return None
    loop.log(f"gate {who}: {len(tasks)} exam games on {len(lite)} starts under {cfg} (draws {js[0] if js else 0}-{js[-1] if js else draws - 1})")
    stopped = play_tasks(loop, tasks, games, who_dir, procs, k, rule["prior"], f"gate-{who}", steps="fight_exam:exam_game_steps", daemon=daemon)
    n = stamp_capped(games, tasks, lite, eng, cfg)
    if n:
        loop.log(f"gate {who}: {n} capped game{'s' if n > 1 else ''} (an engine fault, a hang or a dead worker) counted as lost on engine {eng}")
    return stopped


def stamp_capped(games: Path, tasks: list, lite: dict, eng: str, cfg: str) -> int:
    """A game the lockstep's guard capped (an engine fault, a hang past the deadline, a dead worker) is written without the game's own fields:
    no engine, no config, no start HP. It is a lost game of this engine (`capped: lost`): append a copy with those fields (start HP from the
    start's record, end HP 0, `stamped`) so the pairing and the HP reading count it and the game is not played again at every gate (the last
    row of a key wins; the bare row stays). Only the keys of `tasks` that have no row on an engine are stamped. Returns how many."""
    want = {(t[1], int(t[2])) for t in tasks}
    rows = fg.read_jsonl(games)
    have = {(r["start"], int(r["j"])) for r in rows if r.get("engine") and "start" in r and "j" in r}
    add = []
    for r in rows:
        if "start" not in r or "j" not in r or r.get("engine") or not r.get("capped"):
            continue
        key = (r["start"], int(r["j"]))
        if key not in want or key in have:
            continue
        s = lite.get(key[0]) or {}
        hp0 = int(s.get("hp") or 0)
        add.append({**r, "engine": eng, "config": cfg, "won": False, "capped": True, "start_hp": hp0, "end_hp": 0, "hp_lost": hp0,
                    "cell": r.get("cell") or s.get("cell"), "stratum": r.get("stratum") or s.get("stratum"),
                    "stamped": "a capped row without the game's fields: a loss on this engine"})
        have.add(key)
    if add:
        end = games.read_bytes()[-1:] if games.is_file() else b"\n"
        with open(games, "a", encoding="utf-8", newline="\n") as fh:
            fh.write("" if end in (b"\n", b"") else "\n")
            fh.writelines(json.dumps(x) + "\n" for x in add)
    return len(add)


def cost_by_cell(rows, cells: dict[str, str]) -> dict:
    """CPU-seconds per game by cell, from the rows' own `seconds` (a game's own compute in the lockstep), and the total."""
    acc: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        if r.get("seconds") is not None and r.get("start") in cells:
            acc[cells[r["start"]]].append(float(r["seconds"]))
    return {"by_cell": {c: {"games": len(v), "s_per_game": round(sum(v) / len(v), 2)} for c, v in sorted(acc.items())},
            "games": sum(len(v) for v in acc.values()), "cpu_s": round(sum(sum(v) for v in acc.values()), 1)}


def act3_gate_ids(exam: Path, rule: dict) -> list[str]:
    """The Act 3 guard's start ids: `exam/starts-act3-gate.jsonl` if it is there, else drawn once (`gym_gate.pick_act3_gate`, the seeded
    draw of the rule) from `exam/starts-act3.lite.json` and written there. [] when the exam has no Act 3 cells."""
    path = exam / "starts-act3-gate.jsonl"
    if path.is_file():
        return [r["id"] for r in fg.read_jsonl(path)]
    lite = read_json(exam / "starts-act3.lite.json", None) or []
    if not lite:
        return []
    g = rule["act3_guard"]
    ids = gg.pick_act3_gate(lite, int(g["starts"]), {str(b): int(n) for b, n in g["quotas"].items()}, int(g["seed"]))
    cell = {s["id"]: s["cell"] for s in lite}
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text("".join(json.dumps({"id": i, "cell": cell[i]}) + "\n" for i in ids), encoding="utf-8", newline="\n")
    os.replace(tmp, path)
    return ids


def act3_guard_since(loop: Loop) -> str | None:
    """When the daemon's first practice segment that drew Act 3 starts finished (`games/segments.jsonl`: `cats` has an act3 cell), or None."""
    for r in fg.read_jsonl(loop.games / "segments.jsonl"):
        if any(str(c).startswith("act3") and n for c, n in (r.get("cats") or {}).items()):
            return r.get("t")
    return None


def act3_guard_due(loop: Loop, rule: dict, now: dt.datetime | None = None) -> bool:
    """The Act 3 guard applies from `after_h` hours after the first segment with Act 3 starts."""
    t = gg.parse_stamp(act3_guard_since(loop))
    return t is not None and ((now or utc()) - t).total_seconds() / 3600.0 >= float(rule["act3_guard"]["after_h"])


def run_act3_guard(loop: Loop, cand: str, cand_dir: Path, inc: str, inc_dir: Path, eng: str, rule: dict, procs: int, k: int, daemon=None) -> dict:
    """One draw of the Act 3 guard sample for the candidate and the incumbent (each player's rows kept in `exam/act3-gate/`, the
    incumbent's reused and identity-checked as the human stratum's are). {fail, by_cell, ...}, {"stopped": why} when cut short, or
    {"skipped": why}."""
    exam = loop.root / "exam"
    ids = act3_gate_ids(exam, rule)
    lite_all = _fight_exam().load_act3_exam(exam)
    lite = {i: lite_all[i] for i in ids if i in lite_all}
    if not lite:
        return {"skipped": "no Act 3 exam cells in the exam folder"}
    starts_file = _fight_exam().exam_paths(exam)["act3_starts"]
    cells = {i: s["cell"] for i, s in lite.items()}
    draws = int(rule["act3_guard"]["draws"])
    (exam / "act3-gate").mkdir(exist_ok=True)
    sets = {}
    for who, wdir in ((inc, inc_dir), (cand, cand_dir)):
        games = exam / "act3-gate" / f"games-{who}.jsonl"
        ok = ensure_identity(loop, who, wdir, games, starts_file, lite, eng, rule, procs, k, daemon, tag=f"{cand} Act 3 guard", sub="act3")
        if ok == "stopped":
            return {"stopped": "stopped"}
        stopped = exam_play(loop, who, wdir, games, starts_file, lite, eng, ok, draws, rule, procs, k, daemon)
        if stopped:
            return {"stopped": stopped}
        sets[who] = usable_index(exam_rows(games, lite, rule["config"]), eng, ok, lambda j: j < draws)
    g = gg.act3_guard(sets[cand], sets[inc], rule, cells)
    g["cost"] = cost_by_cell(sets[cand].values(), cells)
    return g


def v4_cells(human: dict) -> tuple[dict, dict]:
    """({start id: the act-and-kind cell, e.g. act1-boss}, {start id: the boss cell, e.g. act1-boss-74}) of the human stratum."""
    kind = {sid: s.get("kind_cell") or "-".join(s["cell"].split("-")[:2]) for sid, s in human.items()}
    boss = {sid: s["cell"] for sid, s in human.items() if kind[sid].endswith("boss")}
    return kind, boss


def _with_human(by: dict, keys: list, cellmap: dict, human: dict) -> dict:
    """`gym_gate.exam_read`'s by-cell figures with the humans' own survival (in 100) and HP lost on the same starts added."""
    out = {}
    for cell, d in by.items():
        ids = sorted({s for s, _j in keys if cellmap.get(s) == cell})
        out[cell] = {**d, "human_surv": round(100 * sum(1.0 if human[s].get("human_survived") else 0.0 for s in ids) / len(ids), 1),
                     "human_hp_lost": round(sum(float(human[s].get("human_hp_lost") or 0.0) for s in ids) / len(ids), 2)}
    return out


def v4_human_reading(fe, human: dict, ci: dict, ii: dict, inc: str, cand: str, draws: int, cfg: str, read: dict) -> dict:
    """The human-bar line of a v4 stage-2 or stage-3 verdict, from the exam games the verdict stands on (no game is played): the exam's
    own reading (`fight_exam.human_bar_reading`, bootstrap interval) and its line, with survival and HP lost by cell (the six act-and-kind
    cells, and each boss) of the candidate, the incumbent and the human."""
    reading = fe.human_bar_reading(ci, ii, human, draws=draws)
    reading.update(incumbent=inc, candidate=cand, config=cfg)
    reading["line"] = fe.human_bar_line(reading, inc)
    keys = sorted(set(ci) & set(ii))
    kind, boss = v4_cells(human)
    reading["by_cell"] = _with_human(read.get("by_cell") or {}, keys, kind, human)
    reading["by_boss"] = _with_human(gg.exam_read(ci, ii, boss)["by_cell"], keys, boss, human)
    return reading


def _v4_note(v: dict) -> str:
    if v.get("verdict") == "incomplete" or v.get("win_diff") is None:
        return ""
    hp = f"; HP lost a fight {v['hp_diff']:+.2f} (SE {v['hp_se']:.2f})" if v.get("hp_diff") is not None and v.get("hp_se") is not None else ""
    if v.get("stage") == 1:
        return f"; 95% upper bound {100 * (v.get('ub95') or 0.0):+.2f} points" + hp
    return f"; 90% lower bound {100 * (v.get('lb90') or 0.0):+.2f} points" + hp + (f"; {v['why']}" if v.get("why") else "")


def run_gate_v4(loop: Loop, cand: str, cand_dir: Path, procs: int, k: int, daemon=None, rule: dict | None = None,
                now: dt.datetime | None = None) -> dict:
    """Score a checkpoint against the incumbent with rule v4 (resumable: the exam games on disk are kept). Plays only the candidate's
    games and, the first time, the incumbent's missing ones. Returns the verdict dict (`verdict` promote, reject or stopped; `stage` 1 to 3)
    with `human` (the human-bar line, stages 2 and 3), `cost` and, when a would-be promotion met the Act 3 guard, `act3`."""
    rule = dict(rule or PROMOTION_RULE_V4)
    fe = _fight_exam()
    exam = loop.root / "exam"
    human = fe.load_human_exam(exam)
    inc, inc_dir = loop.incumbent()
    eng = engine_tag()
    cfg = rule["config"]
    out: dict = {"candidate": cand, "incumbent": inc, "config": cfg, "engine": eng, "rule": rule["name"]}
    if not human:
        return {**out, "verdict": "stopped", "why": "no human stratum in the exam folder (fight_exam.py add-human)"}
    rule["starts"] = len(human)
    cells = v4_cells(human)[0]
    starts_file = fe.exam_paths(exam)["human_starts"]
    games = {inc: fe.games_path(exam, inc), cand: fe.games_path(exam, cand)}
    dirs = {inc: inc_dir, cand: cand_dir}
    oks: dict[str, set] = {}
    for who in (inc, cand):
        oks[who] = ensure_identity(loop, who, dirs[who], games[who], starts_file, human, eng, rule, procs, k, daemon, tag=cand)
        if oks[who] == "stopped":
            return {**out, "verdict": "stopped", "why": "stopped"}
    for stage, draws in enumerate(rule["stage_draws"], 1):
        for who in (inc, cand):
            stopped = exam_play(loop, who, dirs[who], games[who], starts_file, human, eng, oks[who], draws, rule, procs, k, daemon)
            if stopped:
                return {**out, "verdict": "stopped", "why": stopped}
        want = lambda j, d=draws: j < d  # noqa: E731
        crows, irows = exam_rows(games[cand], human, cfg), exam_rows(games[inc], human, cfg)
        ci, ii = usable_index(crows, eng, oks[cand], want), usable_index(irows, eng, oks[inc], want)
        v = gg.v4_decide(ci, ii, stage, rule, cells)
        v["draws"] = draws
        out.update(v)
        if v["verdict"] == "incomplete":
            break
        v_cost = cost_by_cell([r for key, r in ci.items()], cells)
        out["cost"] = v_cost
        if stage >= 2:
            out["human"] = v4_human_reading(fe, human, ci, ii, inc, cand, draws, cfg, v)
            loop.log(f"gate {cand}: {out['human']['line']}")
        loop.log(f"gate {cand}: stage {stage} (draws 0-{draws - 1}) {v['verdict']}: cand {100 * v['cand_rate']:.1f} inc {100 * v['inc_rate']:.1f} "
                 f"of {v['pairs']} games on {v['starts']} starts ({100 * v['win_diff']:+.2f} points, SE {100 * (v['win_se'] or 0):.2f})" + _v4_note(v))
        if v["verdict"] != "continue":
            break
    g = rule["act3_guard"]
    if out["verdict"] == "promote" and g.get("when") == "promote" and act3_guard_due(loop, rule, now):
        a3 = run_act3_guard(loop, cand, cand_dir, inc, inc_dir, eng, rule, procs, k, daemon)
        if "stopped" in a3:
            return {**out, "verdict": "stopped", "why": a3["stopped"]}
        out["act3"] = a3
        if a3.get("fail"):
            out.update(verdict="reject", path=None, why=a3["why"], guard_act3=True)
        loop.log(f"gate {cand}: Act 3 guard: " + (a3.get("skipped") or f"{100 * a3['cand_rate']:.1f} against {100 * a3['inc_rate']:.1f} on {a3['pairs']} games "
                 f"({100 * a3['win_diff']:+.2f} points, 95% upper bound {100 * (a3['ub95'] or 0):+.2f}, floor {g['points']:+.1f}): "
                 + ("FAILED, the promotion is refused" if a3["fail"] else "passed")))
    elif out["verdict"] == "promote":
        loop.log(f"gate {cand}: Act 3 guard not due" + (" (no practice segment has drawn Act 3 starts yet)" if act3_guard_since(loop) is None
                                                         else f" (the first Act 3 practice segment ended {act3_guard_since(loop)}; due {g['after_h']:g} h later)"))
    return out


def import_games(root: Path, name: str, src: Path, provenance: str) -> int:
    """Append the rows of `src` that `exam/games-<name>.jsonl` lacks (by start and draw; the row's own engine tag stays) with a
    `provenance` field, for a candidate or an incumbent whose exam games were played elsewhere (D57 addendum 9: c0380's and v3's of
    2026-11-20). Returns the rows added. A row of the file is never changed."""
    fe = _fight_exam()
    dst = fe.games_path(Path(root) / "exam", name)
    have = {(r["start"], int(r["j"])) for r in fg.read_jsonl(dst)}
    take = [{**r, "provenance": provenance} for r in fg.read_jsonl(src) if (r["start"], int(r["j"])) not in have]
    if take:
        end = dst.read_bytes()[-1:] if dst.is_file() else b"\n"
        with open(dst, "a", encoding="utf-8", newline="\n") as fh:
            fh.write("" if end in (b"\n", b"") else "\n")
            fh.writelines(json.dumps(r) + "\n" for r in take)
    return len(take)


# ---------------------------------------------------------------------------------------------------------- the gate under rule v5


GATE5_DIR = "gate5"
GATE5_STEMS = {"h": "starts-gate", "a3": "starts-gate-act3"}


def gate5_paths(root: Path) -> dict:
    """Rule v5's ruler folder: `gate5/` with a start file (and its index and lite list) per part, `meta.json`, the players' games
    (`games-<name>.jsonl`, both parts) and the identity ledger."""
    d = Path(root) / GATE5_DIR
    out = {"dir": d, "meta": d / "meta.json"}
    for part, stem in GATE5_STEMS.items():
        out[part] = d / f"{stem}.jsonl"
        out[f"{part}_lite"] = d / f"{stem}.lite.json"
    return out


def load_gate5(root: Path) -> dict[str, dict]:
    """{part: {start id: lite record}} of the v5 ruler (empty parts when `gate5-set` has not been run)."""
    p = gate5_paths(root)
    return {part: {s["id"]: s for s in (read_json(p[f"{part}_lite"], None) or [])} for part in GATE5_STEMS}


def gate5_cells(part: str, lite: dict) -> dict[str, str]:
    """Start id -> the cell its weight is read under: the act-and-kind cell in part h (act1-boss, ...), the boss and turn in part a3."""
    if part == "h":
        return {sid: s.get("kind_cell") or f"act{int(s['act'])}-{s['kind']}" for sid, s in lite.items()}
    return {sid: s["cell"] for sid, s in lite.items()}


def gate5_era(loop: Loop) -> int:
    """The era: how many promotions rule v5 has made (`state.json` `gate5.era`); era e plays draws `era_draws` x e onwards."""
    return int((loop.state.get("gate5") or {}).get("era", 0))


def era_js(rule: dict, part: str, era: int, stage: int) -> tuple[int, ...]:
    """The draws a part plays at a stage of an era: era e's first `stage_draws[stage - 1]` draws, from `era_draws` x e."""
    p = rule["parts"][part]
    base = int(p["era_draws"]) * int(era)
    return tuple(range(base, base + int(p["stage_draws"][stage - 1])))


def v5_rule_line(loop: Loop, rule: dict) -> str:
    """The start-up line's account of rule v5: the ruler, the routes, the era."""
    sets = load_gate5(loop.root)
    r = rule["routes"]
    return (f"its own ruler (gate5/: part h {len(sets['h'])} human starts of Acts 1-2 outside the exam, part a3 {len(sets['a3'])} Act 3 boss starts) "
            f"under {rule['config']} ({rule['prior']} prior), stages of {list(rule['parts']['h']['stage_draws'])} and {list(rule['parts']['a3']['stage_draws'])} "
            f"draws; route F at {r['F']['points']:+g} points (part h), route A3 at {r['A3']['points']:+g} (part a3), each with its 97.5% lower bound "
            f"above 0, HP lost not worse by {r['F']['hp_worse']:g} and the other part not clearly worse; era {gate5_era(loop)}"
            + ("" if all(sets.values()) else "; NO RULER YET: run `gym_loop.py gate5-set` (the gate stops until it exists)"))


def _v5_part_line(p: str, x: dict) -> str:
    if x.get("win_diff") is None:
        return f"part {p}: no pairs"
    return (f"part {p} {100 * x['cand_rate']:.1f} / {100 * x['inc_rate']:.1f} ({100 * x['win_diff']:+.2f} points, SE {100 * (x['win_se'] or 0):.2f}; "
            f"HP {x['hp_diff']:+.2f}) on {x['pairs']} games")


def v5_human(fe, sets: dict, idx: dict, inc: str, cand: str, cfg: str, draws: int, read_h: dict) -> dict:
    """The human line of a v5 verdict from the games it stands on (no game is played): part h's human starts as the exam's reading reads
    them (`fight_exam.human_bar_reading`, its line, by act-and-kind cell and by boss), and part a3's human openings (candidate, incumbent and
    human survival in 100)."""
    human = sets["h"]
    ci, ii = idx["h"]
    reading = fe.human_bar_reading(ci, ii, human, draws=draws)
    reading.update(incumbent=inc, candidate=cand, config=cfg, ruler="gate5 part h")
    reading["line"] = fe.human_bar_line(reading, inc)
    keys = sorted(set(ci) & set(ii))
    kind, boss = v4_cells(human)
    reading["by_cell"] = _with_human(read_h.get("by_cell") or {}, keys, kind, human)
    reading["by_boss"] = _with_human(gg.exam_read(ci, ii, boss)["by_cell"], keys, boss, human)
    ca, ia = idx["a3"]
    hk = [k for k in sorted(set(ca) & set(ia)) if sets["a3"].get(k[0], {}).get("stratum") == "human"]
    if hk:
        ids = sorted({s for s, _j in hk})
        reading["a3_human"] = {"starts": len(ids), "games": len(hk), "cand_surv": round(100 * sum(gg.win(ca[k]) for k in hk) / len(hk), 1),
                               "inc_surv": round(100 * sum(gg.win(ia[k]) for k in hk) / len(hk), 1),
                               "human_surv": round(100 * sum(1.0 if sets["a3"][s].get("human_survived") else 0.0 for s in ids) / len(ids), 1)}
    return reading


def run_gate_v5(loop: Loop, cand: str, cand_dir: Path, procs: int, k: int, daemon=None, rule: dict | None = None) -> dict:
    """Score a checkpoint against the incumbent with rule v5 on its own ruler (`gate5/`; resumable: the games on disk are kept). Plays the
    candidate's games of this era's draws and, once an era, the incumbent's. Returns the verdict (`verdict` promote, reject, incomplete or
    stopped; `stage` 1 to 3; `path` the route; `parts` the two readings; `era`; the top-level survival figures are part h's) with `human`
    at stages 2 and 3 and `cost`."""
    rule = dict(rule or PROMOTION_RULE_V5)
    fe = _fight_exam()
    paths = gate5_paths(loop.root)
    sets = load_gate5(loop.root)
    inc, inc_dir = loop.incumbent()
    eng = engine_tag()
    era = gate5_era(loop)
    cfg = rule["config"]
    out: dict = {"candidate": cand, "incumbent": inc, "config": cfg, "engine": eng, "rule": rule["name"], "era": era}
    if not all(sets.values()):
        return {**out, "verdict": "stopped", "why": "no v5 ruler in gate5/ (gym_loop.py gate5-set)"}
    parts = rule["parts"]
    cells = {p: gate5_cells(p, sets[p]) for p in parts}
    games = {inc: paths["dir"] / f"games-{inc}.jsonl", cand: paths["dir"] / f"games-{cand}.jsonl"}
    dirs = {inc: inc_dir, cand: cand_dir}
    oks: dict[tuple[str, str], set] = {}
    for p in parts:
        for who in (inc, cand):
            ok = ensure_identity(loop, who, dirs[who], games[who], paths[p], sets[p], eng, rule, procs, k, daemon, tag=cand, sub=p,
                                 folder=paths["dir"])
            if ok == "stopped":
                return {**out, "verdict": "stopped", "why": "stopped"}
            oks[(p, who)] = ok
    stages = max(len(x["stage_draws"]) for x in parts.values())
    for stage in range(1, stages + 1):
        js = {p: era_js(rule, p, era, stage) for p in parts}
        for p in parts:
            for who in (inc, cand):
                stopped = exam_play(loop, who, dirs[who], games[who], paths[p], sets[p], eng, oks[(p, who)], js[p][-1] + 1, rule, procs, k,
                                    daemon, js=js[p])
                if stopped:
                    return {**out, "verdict": "stopped", "why": stopped}
        idx, reads, short = {}, {}, []
        for p in parts:
            want = lambda j, s=frozenset(js[p]): j in s  # noqa: E731
            ci = usable_index(exam_rows(games[cand], sets[p], cfg), eng, oks[(p, cand)], want)
            ii = usable_index(exam_rows(games[inc], sets[p], cfg), eng, oks[(p, inc)], want)
            idx[p] = (ci, ii)
            reads[p] = gg.strat_read(ci, ii, cells[p], parts[p]["weights"])
            if reads[p]["pairs"] < len(sets[p]) * len(js[p]):
                short.append(f"part {p} {reads[p]['pairs']} of {len(sets[p]) * len(js[p])} pairs")
        h = reads["h"]
        out.update(stage=stage, draws={p: [js[p][0], js[p][-1]] for p in parts}, parts=reads, pairs=h["pairs"], starts=h["starts"],
                   cand_wins=h["cand_wins"], inc_wins=h["inc_wins"], win_diff=h["win_diff"], win_se=h["win_se"], hp_diff=h["hp_diff"],
                   hp_se=h["hp_se"], cand_rate=h["cand_rate"], inc_rate=h["inc_rate"])
        if short:
            out.update(verdict="incomplete", path=None, why="; ".join(short))
            loop.log(f"gate {cand}: stage {stage} (era {era}) incomplete: {out['why']}")
            break
        v = gg.v5_decide(reads, stage, rule)
        out.update(verdict=v["verdict"], path=v["path"], why=v["why"], worse=v["worse"], routes=v["routes"])
        out["cost"] = {p: cost_by_cell(idx[p][0].values(), cells[p]) for p in parts}
        if stage >= 2:
            try:
                out["human"] = v5_human(fe, sets, idx, inc, cand, cfg, len(js["h"]), h)
                loop.log(f"gate {cand}: {out['human']['line']}")
            except Exception as error:  # noqa: BLE001 -- a report-only reading must never change or lose the verdict
                out["human"] = f"error: {type(error).__name__}"
                loop.log(f"gate {cand}: human bar: {type(error).__name__}: {error}")
        loop.log(f"gate {cand}: stage {stage} (era {era}; part h draws {js['h'][0]}-{js['h'][-1]}, part a3 draws {js['a3'][0]}-{js['a3'][-1]}) "
                 f"{v['verdict']}{' by route ' + v['path'] if v['path'] else ''}: " + "; ".join(_v5_part_line(p, reads[p]) for p in parts)
                 + (f"; {v['why']}" if v.get("why") else ""))
        if v["verdict"] != "continue":
            break
    return out


def build_gate5(root: Path, *, seed: int | None = None, rebuild: bool = False, rule: dict | None = None) -> dict:
    """Rule v5's ruler, written once to `gate5/` (returns its meta). Part h: human starts of Acts 1 and 2 from the loop's human start list
    (`starts/human-starts.*`, the corpus's training fifths), only those with the human's result, whose run the loop never played and that the
    fight exam's exclusion list does not hold (so none of the exam's 400), at most one start a run, `set.h_quota` a cell by the seeded hash.
    Part a3: lane 5's Act 3 cells and quotas (`set.a3_*_quota`) drawn from the Act 3 start lists the same way (the exam's own Act 3 starts,
    which the exclusion list holds, are left out; human openings whose run the practice never played first), or, with `set.act3_source`
    "exam", the exam's own Act 3 set copied. The exam fifth: the start list was built without it, and the fixed hash rule is checked again
    here (the ids file is not read). The ruler's runs and starts are added to `exam/exclude_runs.json` (the old file kept once as
    `.pre-gate5.bak`) so the practice never draws them. An existing ruler is kept unless `rebuild` (its files then get a dated backup name)."""
    import act3_starts as a3  # noqa: PLC0415 -- pure Python
    import human_starts as hs  # noqa: PLC0415 -- pure Python; the corpus is not read here

    rule = rule or PROMOTION_RULE_V5
    st = rule["set"]
    seed = int(st["seed"] if seed is None else seed)
    root = Path(root)
    fe = _fight_exam()
    paths = gate5_paths(root)
    if paths["meta"].exists() and not rebuild:
        raise FileExistsError(f"{paths['meta']} exists: the ruler is fixed; --rebuild replaces it")
    sd = root / "starts"
    exam = fe.exam_paths(root / "exam")
    played = fe.played_runs(root)
    ex = pmix.read_exclusion(root)
    # part h
    lite = read_json(sd / f"{hs.STEM}.lite.json", []) or []
    outcomes = (read_json(sd / hs.OUTCOME_FILE, {}) or {}).get("outcomes") or {}
    if not lite or not outcomes:
        raise FileNotFoundError(f"no human starts or no outcomes in {sd}")
    left: Counter = Counter()
    pool: dict[str, list[dict]] = defaultdict(list)
    for s in lite:
        if s["id"] not in outcomes:
            left["no outcome in the sidecar"] += 1
        elif {pmix.run_of(s["id"]), s.get("run"), s.get("seed")} & played:
            left["its run was played by the loop"] += 1
        elif pmix.excluded(s, ex):
            left["on the exam's exclusion list"] += 1
        else:
            pool[fe.cell_of(s)].append(s)
    taken: set = set()
    chosen: list[dict] = []
    rep_h: dict[str, dict] = {}
    for c in sorted(st["h_quota"]):
        q, took = int(st["h_quota"][c]), 0
        for s in sorted(pool.get(c, []), key=lambda s: fe.order_key(seed, s["id"])):
            if took >= q:
                break
            if s["run"] in taken:
                continue
            taken.add(s["run"])
            took += 1
            chosen.append({**s, "cell": c, "kind_cell": fe.kind_cell(s), "half": fe.half_of(s["id"]), "practised": 0, "clean": True, "stratum": "human"})
        rep_h[c] = {"pool": len(pool.get(c, [])), "quota": q, "chosen": took, "short": took < q}
    hs.assert_no_exam((s["run"] for s in chosen), frozenset())  # the fixed hash rule of D38 (the list itself was built without the fifth)
    src = str(sd / f"{hs.STEM}.jsonl")
    keys = ("cell", "kind_cell", "half", "practised", "clean", "stratum")
    full_h = [{**load_start(src, s["id"]), **{k: s[k] for k in keys}, **outcomes[s["id"]]} for s in chosen]
    # part a3
    rep_a: dict = {}
    if st.get("act3_source", "twin") == "exam":
        ids = [s["id"] for s in (read_json(exam["act3_lite"], None) or [])]
        full_a = [load_start(str(exam["act3_starts"]), i) for i in ids]
        rep_a = {"source": "the exam's own Act 3 set", "starts": len(full_a)}
    else:
        h3 = read_json(sd / f"{a3.HUMAN_STEM}.lite.json", []) or []
        o3 = read_json(sd / f"{a3.OWN_STEM}.lite.json", []) or []
        out3 = (read_json(sd / a3.OUTCOME_FILE, {}) or {}).get("outcomes") or {}
        if not h3 or not out3:
            raise FileNotFoundError(f"no human Act 3 starts or no outcomes in {sd}")

        def skip(r: dict, human: bool) -> str | None:
            if human and r["id"] not in out3:
                return "no outcome in the sidecar"
            if pmix.excluded(r, ex):
                return "on the exam's exclusion list"
            if {pmix.run_of(r["id"]), r.get("run")} & played:
                return "practised by the loop"
            return None

        taken3: set = set()
        ch, rh = a3.pick_quota(h3, st["a3_human_quota"], seed=seed, taken=taken3, skip=lambda r: skip(r, True),
                               prefer=lambda r: not r.get("loop_played"))
        co, ro = a3.pick_quota(o3, st["a3_own_quota"], seed=seed, taken=taken3, skip=lambda r: skip(r, False)) if o3 else ([], {})
        full_a = []
        for stem, picked in ((a3.HUMAN_STEM, ch), (a3.OWN_STEM, co)):
            src3 = str(sd / f"{stem}.jsonl")
            for s in picked:
                doc = load_start(src3, s["id"])
                extra = {"cell": s["cell"], "kind_cell": fe.kind_cell(doc), "half": fe.half_of(s["id"]), "practised": 0 if s["clean"] else 1,
                         "clean": s["clean"], "stratum": doc.get("stratum") or ("human" if stem == a3.HUMAN_STEM else "own")}
                if stem == a3.HUMAN_STEM:
                    extra.update(out3[s["id"]])
                full_a.append({**doc, **extra})
        rep_a = {"source": "lane 5's design, drawn again outside the exam", "human": rh, "own": ro}
    if not full_h or not full_a:
        raise ValueError(f"an empty part: h {len(full_h)}, a3 {len(full_a)}")
    paths["dir"].mkdir(parents=True, exist_ok=True)
    if rebuild:
        tag = stamp().replace(":", "")
        for f in list(paths["dir"].glob("starts-gate*")) + [paths["meta"]]:
            if f.is_file():
                f.rename(f.with_name(f.name + f".{tag}.bak"))
    fe.write_stem_set(paths["dir"], GATE5_STEMS["h"], full_h)
    fe.write_stem_set(paths["dir"], GATE5_STEMS["a3"], full_a)
    # the practice never draws the ruler's starts or their runs
    exf = exam["exclude"]
    exd = read_json(exf, {}) or {}
    backup = exf.with_name(exf.name + ".pre-gate5.bak")
    if exf.exists() and not backup.exists():
        backup.write_bytes(exf.read_bytes())
    runs = sorted({str(s["run"]) for s in full_h + full_a if s.get("run")})
    before = len(set(exd.get("run_seeds") or []))
    exd["run_seeds"] = sorted(set(exd.get("run_seeds") or []) | set(runs))
    exd["start_ids"] = sorted(set(exd.get("start_ids") or []) | {s["id"] for s in full_h + full_a})
    exd["by_stratum"] = {**(exd.get("by_stratum") or {}), "gate5-h": len(full_h), "gate5-a3": len(full_a)}
    exd["gate5"] = {"rule": rule["name"], "written": stamp(), "starts": len(full_h) + len(full_a), "runs": len(runs)}
    write_json(exf, exd)
    by_part = {"h": dict(sorted(Counter(s["cell"] for s in full_h).items())), "a3": dict(sorted(Counter(s["cell"] for s in full_a).items()))}
    meta = {"rule": rule["name"], "written": stamp(), "seed": seed, "starts": {"h": len(full_h), "a3": len(full_a)}, "by_part": by_part,
            "by_stratum_a3": dict(Counter(s.get("stratum") for s in full_a)), "report_h": rep_h, "report_a3": rep_a, "left_out_h": dict(left),
            "short_cells": sorted(c for c, x in rep_h.items() if x["short"]) + sorted(
                c for side in ("human", "own") for c, x in ((rep_a.get(side) or {}).items()) if isinstance(x, dict) and x.get("short")),
            "played_runs": len(played), "weights": {p: dict(x["weights"]) for p, x in rule["parts"].items()},
            "human_survived": {"h": sum(1 for s in full_h if s.get("human_survived")),
                               "a3": sum(1 for s in full_a if s.get("stratum") == "human" and s.get("human_survived"))},
            "exclusion": {"file": exf.name, "runs_added": len(set(exd["run_seeds"])) - before, "backup": backup.name},
            "files_sha256": {f.name: sha256_file(f) for f in (paths["h"], paths["a3"])},
            "note": "the exam's 400 human starts and its 414 Act 3 starts are not in this ruler; the corpus's exam fifth never is"}
    write_json(paths["meta"], meta)
    loop_log = Path(root) / "loop.log"
    with open(loop_log, "a", encoding="utf-8") as fh:
        fh.write(f"{stamp()} gate5-set: rule v5's ruler written: part h {len(full_h)} starts, part a3 {len(full_a)}; "
                 f"{meta['exclusion']['runs_added']} runs added to the exam's exclusion list\n")
    return meta


# ---------------------------------------------------------------------------------------------------------- the human bar at the gate


HUMAN_BAR = {"name": "gym-loop-human-bar-v1", "config": "a3-v1", "draws": 4, "prior": "cpu"}
# What the gate plays after a stage-2 promotion (`--human-bar promotion`, the default since 2026-11-11; `on`: after every stage-2 verdict, `off`:
# never): the fight exam's human stratum under the exam's own configuration (`fight_exam.DEFAULT_CONFIG`, `DRAWS`), whatever the gate's own
# `--player-config` is, on the CPU prior (the reading the weekly comparison and v2's and v3's were taken with, so the numbers compare).
# Report-only: no part of it enters PROMOTION_RULE or `decide`.


def _fight_exam():
    """`fight_exam`, which imports this module as `gym_loop`: when the daemon runs this file as `__main__` the module is registered under
    its own name first, so the exam shares these functions and constants instead of loading a second copy."""
    sys.modules.setdefault("gym_loop", sys.modules[__name__])
    import fight_exam  # noqa: PLC0415

    return fight_exam


def human_bar(loop: Loop, cand: str, cand_dir: Path, inc: str, procs: int, k: int, prior: str, daemon=None) -> dict | str:
    """The candidate on the fight exam's human stratum, then the human line. Plays the stratum's games the candidate has no row of
    (`exam/games-<cand>.jsonl`, resumable) with the gate's own lockstep play (`play_tasks`: BelowNormal, the QUIET line, the stop
    reasons of `why_blocked` through `daemon`), reads the candidate's and the incumbent's games on it (`exam/games-<inc>.jsonl`; the
    incumbent's side is left out if it has none) and returns the reading (`fight_exam.human_bar_reading` with its `line`), or "stopped"
    when a stop cut the play short, or "no human stratum" when the exam has none."""
    fe = _fight_exam()
    exam = loop.root / "exam"
    human = fe.load_human_exam(exam)
    if not human:
        loop.log(f"gate {cand}: human bar: no human stratum in {exam.name}/ (fight_exam.py add-human); nothing played")
        return "no human stratum"
    games = fe.games_path(exam, cand)
    cfg, draws = HUMAN_BAR["config"], int(HUMAN_BAR["draws"])
    done = {(r["start"], int(r["j"])) for r in fg.read_jsonl(games)}
    tasks = fe.exam_tasks({}, cand, cand, cfg, exam, done, draws=draws, human=human)
    if tasks:
        loop.log(f"gate {cand}: human bar: {cand} plays {len(tasks)} games on {len(human)} human-stratum starts under {cfg}")
        stopped = play_tasks(loop, tasks, games, cand_dir, procs, k, prior, f"human-bar-{cand}", steps="fight_exam:exam_game_steps",
                             daemon=daemon)
        if stopped:
            loop.log(f"gate {cand}: human bar: stopped ({stopped}); the games played stay in {games.name}")
            return "stopped"
    inc_games = fe.human_games(fg.read_jsonl(fe.games_path(exam, inc)), human, cfg)
    reading = fe.human_bar_reading(fe.human_games(fg.read_jsonl(games), human, cfg), inc_games, human, draws=draws)
    reading.update(incumbent=inc, candidate=cand, config=cfg)
    reading["line"] = fe.human_bar_line(reading, inc)
    loop.log(f"gate {cand}: {reading['line']}")
    return reading


def human_bar_adopt(loop: Loop, cand: str, ver: str) -> int:
    """A promoted candidate's human games are found under the new version's name: `exam/games-<ver>.jsonl` gets the rows of
    `exam/games-<cand>.jsonl` it lacks, relabelled as `reuse_incumbent_games` relabels the gate's (the candidate's file stays).
    Returns how many rows were added."""
    exam = loop.root / "exam"
    src, dst = exam / f"games-{cand}.jsonl", exam / f"games-{ver}.jsonl"
    if not src.is_file():
        return 0
    have = {(r["start"], int(r["j"])) for r in fg.read_jsonl(dst)}
    take = [{**r, "arm": ver, "player": ver, "reused_from": cand} for r in fg.read_jsonl(src) if (r["start"], int(r["j"])) not in have]
    if take:
        end = dst.read_bytes()[-1:] if dst.is_file() else b"\n"
        with open(dst, "a", encoding="utf-8", newline="\n") as fh:
            fh.write("" if end in (b"\n", b"") else "\n")
            fh.writelines(json.dumps(r) + "\n" for r in take)
        loop.log(f"gate {cand}: human bar: {len(take)} human-stratum games of {cand} kept as {ver}'s ({dst.name})")
    return len(take)


# ---------------------------------------------------------------------------------------------------------- the gate's pre-filter


PREFILTER_SCORE_TIMEOUT_S = 900.0
_SHA_CACHE: dict = {}


def _t(s: str) -> dt.datetime:
    return dt.datetime.strptime(s, "%Y-%m-%dT%H:%M:%SZ").replace(tzinfo=dt.timezone.utc)


def policy_sha(d: Path) -> str:
    """sha256 of a folder's combat_policy.pt (kept by path, size and time: an incumbent's file does not change)."""
    p = Path(d) / "combat_policy.pt"
    s = p.stat()
    key = (str(p), s.st_mtime_ns, s.st_size)
    if key not in _SHA_CACHE:
        _SHA_CACHE[key] = sha256_file(p)
    return _SHA_CACHE[key]


def prefilter_cfg(a: argparse.Namespace | None = None) -> dict:
    """PREFILTER's settings, overridden by the daemon's flags (`--prefilter`, `--prefilter-margin`, `--prefilter-floor-h`)."""
    cfg = {**PREFILTER, "window": 300_000, "rule": PROMOTION_RULE["name"]}
    if a is not None:
        cfg.update(mode=a.prefilter, margin=a.prefilter_margin, floor_h=a.prefilter_floor_h, window=a.train_window, rule=active_rule(a)["name"])
        if getattr(a, "gate_every_min", 0.0):
            cfg["gate_every_min"] = float(a.gate_every_min)
    return cfg


def prefilter_state(loop: Loop, now: dt.datetime | None = None) -> dict:
    """The pre-filter's record in state.json: the incumbent it is about, the time of the last gate (the floor counts from it), the
    checkpoints skipped since then, each decided checkpoint's figures and the best gap among those gated and rejected. A new
    incumbent starts a new record: the gaps were measured against the old one."""
    inc = loop.state["incumbent"]
    pf = loop.state.get("prefilter")
    if not isinstance(pf, dict) or pf.get("incumbent") != inc:
        last = next((r["t"] for r in reversed(fg.read_jsonl(loop.gates / "verdicts.jsonl")) if r.get("t")),
                    (now or utc()).strftime("%Y-%m-%dT%H:%M:%SZ"))  # the last gate played, whoever it was against
        pf = {"rule": PREFILTER["name"], "incumbent": inc, "last_gate": last, "window": [], "figures": {}, "best": None}
        loop.state["prefilter"] = pf
    return pf


def _figures_ok(d: dict, inc: str, sha: str, split: str) -> bool:
    """A held-out record is the incumbent's reference only when it names this incumbent (version and weights) and this split."""
    try:
        return (d.get("base_version") == inc and d.get("base_sha256") == sha and d.get("split") == split
                and math.isfinite(float(d["val_policy_ce"])) and math.isfinite(float(d["base_val_policy_ce"])))
    except (KeyError, TypeError, ValueError):
        return False


def _fig(d: dict, source: str) -> dict:
    loss, ref = float(d["val_policy_ce"]), float(d["base_val_policy_ce"])
    return {"loss": loss, "ref": ref, "gap": ref - loss, "rows": d.get("val_rows"), "source": source}


def score_checkpoint(loop: Loop, cand_dir: Path, inc_dir: Path, window: int) -> dict:
    """The held-out figures of a checkpoint and the incumbent on the same rows, from the trainer's `score` command (GPU venv, the
    CPU, one thread); an error raises."""
    argv = [GPU_PYTHON, str(ROOT / "rl" / "gym_loop_train.py"), "score", "--root", str(loop.root), "--candidate", str(cand_dir),
            "--base", str(inc_dir), "--window", str(window)]
    r = subprocess.run(argv, capture_output=True, text=True, timeout=PREFILTER_SCORE_TIMEOUT_S, cwd=str(ROOT),
                       env={**os.environ, "CUDA_VISIBLE_DEVICES": ""}, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0),
                       check=False)
    if r.returncode != 0:
        raise RuntimeError(f"score rc={r.returncode}: {(r.stderr or r.stdout).strip()[-300:]}")
    return json.loads(r.stdout.strip().splitlines()[-1])


def checkpoint_figures(loop: Loop, cand_dir: Path, window: int = 300_000, scorer=None) -> dict:
    """{loss, ref, gap, rows, source}: the checkpoint's held-out policy cross-entropy and the incumbent's on the same rows. They come
    from the checkpoint's train.json when that names this incumbent and split, else from the checkpoint's prefilter.json when that
    does, else the incumbent is scored once on the current held-out rows with the checkpoint (`scorer`) and the record is stored in
    prefilter.json: so a new incumbent or a new split recomputes the reference, and nothing is scored twice."""
    cand_dir = Path(cand_dir)
    inc, inc_dir = loop.incumbent()
    sha, split = policy_sha(inc_dir), val_split_id()
    for name in ("train.json", "prefilter.json"):
        d = read_json(cand_dir / name, {}) or {}
        if _figures_ok(d, inc, sha, split) and (name == "train.json" or d.get("sha256") == policy_sha(cand_dir)):
            return _fig(d, name)
    got = (scorer or score_checkpoint)(loop, cand_dir, inc_dir, window)
    rec = {"checkpoint": cand_dir.name, "t": stamp(), "base_version": inc, **got}
    if not _figures_ok(rec, inc, sha, split):
        raise RuntimeError("the scored figures do not name this incumbent and split")
    write_json(cand_dir / "prefilter.json", rec)
    return _fig(rec, "scored")


def prefilter_decide(pf: dict, fig: dict, cfg: dict) -> tuple[bool, str]:
    """(gate it, why). The gap is the incumbent's held-out loss minus the checkpoint's: it must exceed the margin, and in mode
    'best' also the gap of every checkpoint already gated and rejected against this incumbent (a re-test of a net no better than
    one the gate has refused tells the gate nothing)."""
    gap = fig["gap"]
    if gap <= cfg["margin"]:
        return False, "not below the incumbent's" + (f" by more than {cfg['margin']:.4f}" if cfg["margin"] else "")
    best = pf.get("best")
    if best and best.get("rule", PROMOTION_RULE["name"]) != cfg.get("rule", PROMOTION_RULE["name"]):
        best = None  # D57 addendum 9: a gap anchored by a verdict of another rule (a record with no rule is v3's) does not count
    if cfg["mode"] == "best" and best and gap <= best["gap"]:
        return False, f"not past {best['cand']}'s gap of {best['gap']:+.4f}, which the gate rejected"
    return True, "below the incumbent's" + (f" and past {best['cand']}'s gap of {best['gap']:+.4f}" if cfg["mode"] == "best" and best
                                            else "")


GATE_SLACK_S = 30.0  # the stamps are whole seconds: an interval short of `--gate-every-min` by less than this still counts as passed


def checkpoint_made(d: Path) -> dt.datetime | None:
    """When the trainer wrote a checkpoint: the `t` of its train.json, else its folder's modification time (None: neither is readable)."""
    try:
        return _t(str((read_json(Path(d) / "train.json", {}) or {}).get("t")))
    except (TypeError, ValueError):
        pass
    try:
        return dt.datetime.fromtimestamp(Path(d).stat().st_mtime, dt.timezone.utc)
    except OSError:
        return None


def gate_anchor(loop: Loop) -> dt.datetime | None:
    """When the checkpoint was made that the gate last played to a verdict (a reject or a promotion: not a skip, not a supersede), read
    from the state on disk and the checkpoint's own train.json, so a restart changes nothing. None before the first verdict."""
    done = [n for n, v in loop.state["gated"].items() if isinstance(v, dict) and v.get("verdict") in ("reject", "promote")]
    for n in sorted(done, reverse=True):
        made = checkpoint_made(loop.checkpoints / n)
        if made is not None:
            return made
    return None


def gate_offered(d: Path, anchor: dt.datetime | None, every_min: float) -> bool:
    """Whether a checkpoint is offered to the gate at `--gate-every-min` minutes: it was made that long after the checkpoint the gate last
    played (the trainer still writes one every `--checkpoint-min`; the ones between are never decided, and the next gate supersedes them).
    Zero minutes, or no verdict yet, offers every checkpoint, as before."""
    if not every_min or every_min <= 0 or anchor is None:
        return True
    made = checkpoint_made(d)
    return made is None or (made - anchor).total_seconds() >= every_min * 60 - GATE_SLACK_S


def prefilter_pick(loop: Loop, cfg: dict, now: dt.datetime | None = None, scorer=None) -> tuple[str, Path] | None:
    """The checkpoint to gate now, or None. Every checkpoint without a verdict is decided once (oldest first): one that passes is a
    candidate (the newest passing one is gated, the older ones superseded as before), one that does not is marked skipped, logged
    with both figures and put in the window. The floor: when `floor_h` hours have passed since the last gate, the skipped
    checkpoint with the best gap in the window is gated, so the board keeps moving whatever the figures say. A checkpoint whose
    figures cannot be had goes to the gate (the filter never blocks on its own failure)."""
    now = now or utc()
    pf = prefilter_state(loop, now)
    inc, _ = loop.incumbent()
    gated = loop.state["gated"]
    ungated = sorted(p for p in loop.checkpoints.glob("c[0-9][0-9][0-9][0-9]")
                     if p.name not in gated and (p / "model.json").is_file() and (p / "combat_policy.pt").is_file())
    every = float(cfg.get("gate_every_min") or 0.0)
    if every > 0:  # D60 item 5: only a checkpoint made `every` minutes after the last one gated is decided; the ones between wait
        anchor = gate_anchor(loop)
        ungated = [d for d in ungated if gate_offered(d, anchor, every)]
    chosen = None
    for d in ungated:
        fig = pf["figures"].get(d.name)
        fresh = fig is None
        if fresh:
            try:
                fig = checkpoint_figures(loop, d, int(cfg.get("window", 300_000)), scorer)
            except Exception as error:  # noqa: BLE001 -- logged; the gate decides
                fig = {"error": clean(f"{type(error).__name__}: {error}")[:240]}
        if "error" in fig:
            go, why = True, "no held-out figures, the gate decides"
        else:
            go, why = prefilter_decide(pf, fig, cfg)
        pf["figures"][d.name] = {**fig, "go": go}
        if fresh:
            if "error" in fig:
                loop.log(f"prefilter: {d.name} gated: {why} ({fig['error']})")
            else:
                loop.log(f"prefilter: {d.name} {'gated' if go else 'skipped'}: held-out policy CE {fig['loss']:.4f} against "
                         f"{inc}'s {fig['ref']:.4f} on the same {fig['rows']} rows (gap {fig['gap']:+.4f}): {why}"
                         + ("" if go else f"; {len(pf['window']) + 1} skipped since the last gate"))
        if go:
            chosen = (d.name, d)
        elif d.name not in gated:
            gated[d.name] = {"verdict": "skipped", "why": why, "incumbent": inc, "loss": fig["loss"], "inc_loss": fig["ref"]}
            pf["window"].append(d.name)
    for stale in sorted(pf["figures"])[:-200]:
        pf["figures"].pop(stale, None)
    if chosen is None:
        win = [n for n in pf["window"] if (loop.checkpoints / n).is_dir() and "gap" in pf["figures"].get(n, {})]
        age_h = (now - _t(pf["last_gate"])).total_seconds() / 3600
        if win and age_h >= cfg["floor_h"]:
            n = max(win, key=lambda x: pf["figures"][x]["gap"])
            f = pf["figures"][n]
            loop.log(f"prefilter: floor: {age_h:.1f} h since the last gate, gating {n}, the best of {len(win)} skipped "
                     f"(held-out policy CE {f['loss']:.4f} against {inc}'s {f['ref']:.4f}, gap {f['gap']:+.4f})")
            chosen = (n, loop.checkpoints / n)
    loop.save()
    return chosen


def prefilter_note_verdict(loop: Loop, cand: str, verdict: dict, now: dt.datetime | None = None) -> None:
    """A gate has been played to a verdict: the window is over (the floor counts from now), and the checkpoint's gap joins the
    bar a later checkpoint must pass."""
    pf = prefilter_state(loop, now)
    pf["last_gate"] = (now or utc()).strftime("%Y-%m-%dT%H:%M:%SZ")
    pf["window"] = []
    fig = pf["figures"].get(cand)
    rname = verdict.get("rule") or PROMOTION_RULE["name"]
    if (fig and "gap" in fig and verdict.get("verdict") in ("reject", "promote")
            and (pf["best"] is None or pf["best"].get("rule", PROMOTION_RULE["name"]) != rname or fig["gap"] > pf["best"]["gap"])):
        pf["best"] = {"cand": cand, "gap": fig["gap"], "loss": fig["loss"], "ref": fig["ref"], "rule": rname}


def promote(loop: Loop, cand: str, cand_dir: Path, verdict: dict, board: str) -> str:
    """The checkpoint becomes the next version: copied to players/vN, the incumbent from now on, its board row built."""
    n = 1 + max(int(v[1:]) for v in loop.state["versions"])
    ver = f"v{n}"
    dest = loop.players / ver
    if not dest.exists():
        shutil.copytree(cand_dir, dest)
    loop.state["versions"][ver] = {"dir": str(dest), "source": f"checkpoint {cand}", "n": n, "promoted": stamp(),
                                   "verdict": {k: verdict.get(k) for k in ("cand_wins", "inc_wins", "pairs", "win_diff", "win_se",
                                                                           "l0_diff", "p", "cand_rate", "inc_rate", "path", "diff_lb",
                                                                           "pool_k", "pool_diff", "pool_p")}}
    prev = loop.state["incumbent"]
    loop.state["incumbent"] = ver
    loop.state["promotions"].append({"version": ver, "checkpoint": cand, "over": prev, "t": stamp()})
    loop.state["gate_pair"] = next_gate_pair(PROMOTION_RULE, loop.gate_pair())  # the next candidate meets it on a fresh pair
    loop.save()
    if verdict.get("rule") == PROMOTION_RULE_V5["name"]:
        # v5 plays its own ruler, not the board's starts (no board row); the next era's draws are fresh for the new incumbent
        g5 = loop.state.setdefault("gate5", {"era": 0})
        g5["era"] = int(verdict.get("era") if verdict.get("era") is not None else g5.get("era", 0)) + 1
        g5.setdefault("eras", []).append({"era": g5["era"], "incumbent": ver, "t": stamp(), "over": prev, "route": verdict.get("path")})
        loop.save()
        loop.log(f"promoted {cand} as {ver} over {prev} by rule v5 (route {verdict.get('path')}, stage {verdict.get('stage')}): part h "
                 f"{100 * (verdict.get('win_diff') or 0):+.2f} points, part a3 {100 * ((verdict.get('parts') or {}).get('a3', {}).get('win_diff') or 0):+.2f}; "
                 f"era {g5['era']} begins: {ver} plays its bar on draws nobody has played; no fight-board row")
        return ver
    if verdict.get("rule") == PROMOTION_RULE_V4["name"]:
        # v4 plays the exam, not the board's 200 starts: the candidate has no board games, so the board page gets no row for it here
        loop.log(f"promoted {cand} as {ver} over {prev} by rule v4 (stage {verdict.get('stage')}, {verdict.get('draws')} draws): "
                 f"{verdict.get('cand_wins')} against {verdict.get('inc_wins')} wins of {verdict.get('pairs')} exam games; "
                 f"no fight-board row (the board's starts were not played; play them with `gate --rule v3` to add one)")
        return ver
    rows = fg.read_jsonl(loop.gates / verdict["cand_file"])
    with open(loop.board / f"fp-{ver}.jsonl", "w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps({**{k: v for k, v in r.items() if not k.startswith("_")}, "arm": f"fp-{ver}"}) + "\n")
    loop.log(f"promoted {cand} as {ver} over {prev}: {verdict.get('cand_wins')} against {verdict.get('inc_wins')} wins of "
             f"{verdict.get('pairs')}; stage 1 now plays j {list(stage1_js(PROMOTION_RULE, loop.gate_pair()))}")
    if board != "off":
        try:
            with loop.span("wait", "publish", log=True):  # the board page's build and push: the loop plays nothing meanwhile
                board_commit(loop, ver, verdict, commit=board == "commit")
        except Exception as error:  # noqa: BLE001 -- the board is retried at the next start (`board_pending`)
            loop.log(f"board: {type(error).__name__}: {error}")
    return ver


# ---------------------------------------------------------------------------------------------------------- the fight board

BOARD_REPO = Path(os.environ.get("GYM_LOOP_BOARD_REPO", "D:/spireline/spireline"))
BOARD_WT = Path(os.environ.get("GYM_LOOP_BOARD_WT", "D:/spireline/wt-board"))
BOARD_PAGE = "docs/paper/fight-board.html"
# Where each arm already on the board has its games: the page keeps no games, so a rebuild passes every arm's file again. An arm
# on the page with no known file stops the step (a row is never dropped); `board/games.json` in the loop's folder adds arms.
BOARD_GAMES = {"a3-v1": BOARD_DIR / "games.jsonl", "x16": BOARD_DIR / "games.jsonl", "x4": BOARD_DIR / "games.jsonl",
               "judge-v0": REF / "lab" / "results" / "fight-gym" / "2026-10-30-judge" / "games.jsonl"}
COAUTHOR = "Co-Authored-By: Claude Opus 5.5 <noreply@anthropic.com>"


def _git(*args: str, cwd: Path) -> str:
    r = subprocess.run(["git", *args], cwd=str(cwd), capture_output=True, text=True, encoding="utf-8", errors="replace",
                       check=False, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        raise RuntimeError(clean(f"git {' '.join(args[:2])}: {r.stderr.strip()[-300:]}"))
    return r.stdout


def page_arms(html: str) -> list[tuple[str, str]]:
    """The arms already on the page, in order: (key, label)."""
    out = []
    for m in re.finditer(r'"key":\s*"([^"]+)",\s*"label":\s*"([^"]+)"', html):
        if m.group(1) not in [k for k, _ in out]:
            out.append((m.group(1), m.group(2)))
    return out


def board_args(loop: Loop, arms: list[tuple[str, str]], vers: list[str]) -> tuple[list[str], list[str]]:
    """(--games files, --arm entries): the page's arms with their games, then every promoted version not on it yet."""
    known = {k: Path(v) for k, v in BOARD_GAMES.items()}
    known.update({k: Path(v) for k, v in (read_json(loop.board / "games.json", {}) or {}).items()})
    known.update({f"fp-{v}": loop.board / f"fp-{v}.jsonl" for v in vers})
    keys = [k for k, _ in arms] + [f"fp-{v}" for v in vers if f"fp-{v}" not in [k for k, _ in arms]]
    labels = dict(arms) | {f"fp-{v}": f"Fight player, version {v[1:]}" for v in vers}
    missing = [k for k in keys if k not in known or not known[k].is_file()]
    if missing:
        raise RuntimeError(f"board: no games for the arms {missing}; nothing rebuilt (add them to board/games.json)")
    files = list(dict.fromkeys(str(known[k]) for k in keys))
    return files, [f"{k}={labels[k]}" for k in keys]


def page_log(html: str) -> list[str]:
    m = re.search(r"const LOG = (\[.*?\]);", html, re.S)
    return json.loads(m.group(1)) if m else []


def board_line(ver: str, v: dict, eng_commit: str) -> str:
    """The board's changelog line, in plain language."""
    pts = 100 * (v.get("win_diff") or 0.0)
    return (f"{utc().strftime('%Y-%m-%d')}: fight player, version {ver[1:]} added, trained by the always-on loop on its own "
            f"practice fights; it won {v.get('cand_wins')} of {v.get('pairs')} replays against {v.get('inc_wins')} for the version "
            f"before it on the same fights and draws ({pts:+.1f} points); played on the copy of the game as of {eng_commit}")


def board_commit(loop: Loop, ver: str, verdict: dict, commit: bool = True) -> str:
    """The standing fight-board step: the page rebuilt with every promoted version as a row, committed to Spireline main from its
    own worktree (never the shared checkout), pushed; on a moved main, rebased and pushed again."""
    if not BOARD_WT.exists():
        _git("worktree", "add", "--detach", str(BOARD_WT), "origin/main", cwd=BOARD_REPO)
    _git("fetch", "-q", "origin", cwd=BOARD_WT)
    _git("checkout", "-q", "--detach", "origin/main", cwd=BOARD_WT)
    page = BOARD_WT / BOARD_PAGE
    html = page.read_text(encoding="utf-8")
    old = page_log(html)
    onpage = {k for k, _ in page_arms(html)}
    vers = sorted((v for v in loop.state["versions"] if v != "v0" and (f"fp-{v}" in onpage or (loop.board / f"fp-{v}.jsonl").is_file())),
                  key=lambda x: int(x[1:]))  # a version promoted under rule v4 has no board games and no row
    eng_commit = _git("rev-parse", "--short", "HEAD", cwd=ROOT).strip()
    line = board_line(ver, verdict, eng_commit)
    files, arms = board_args(loop, page_arms(html), vers)
    cmd = [sys.executable, "tools/fight_board.py", "--out", BOARD_PAGE, "--starts", str(BOARD_DIR / "starts.jsonl")]
    for f in files:
        cmd += ["--games", f]
    for a in arms:
        cmd += ["--arm", a]
    for x in [*old, line] if line not in old else old:
        cmd += ["--log", x]
    r = subprocess.run(cmd, cwd=str(BOARD_WT), capture_output=True, text=True, check=False,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    if r.returncode != 0:
        raise RuntimeError(clean("fight_board.py: " + r.stderr[-300:]))
    if not commit:
        shutil.copy(page, loop.board / "fight-board.html")
        return "dry"
    _git("add", BOARD_PAGE, cwd=BOARD_WT)
    v = verdict
    msg = (f"docs: fight board, fight player version {ver[1:]} added (always-on loop: {v.get('cand_wins')} of {v.get('pairs')} "
           f"Act-1 boss replays won against {v.get('inc_wins')}, {100 * (v.get('win_diff') or 0):+.1f} points, McNemar p "
           f"{v.get('p', float('nan')):.3f})\n\n{COAUTHOR}\n")
    _git("commit", "-q", "-m", msg, cwd=BOARD_WT)
    for _ in range(3):
        try:
            _git("push", "-q", "origin", "HEAD:main", cwd=BOARD_WT)
            break
        except RuntimeError:
            _git("fetch", "-q", "origin", cwd=BOARD_WT)
            _git("rebase", "-q", "origin/main", cwd=BOARD_WT)
    else:
        raise RuntimeError("board: push failed three times")
    h = _git("rev-parse", "--short", "HEAD", cwd=BOARD_WT).strip()
    loop.state.setdefault("board", {})[ver] = h
    loop.save()
    loop.log(f"board: {ver} committed {h}")
    return h


# ---------------------------------------------------------------------------------------------------------- the card's split


class GpuWatch:
    """Once a minute, the card's per-process busy share (Windows' GPU Engine counters, the busiest engine per process), split into
    the loop's inference (its prior server), its training (the trainer) and everyone else (the lab's jobs and the desktop). Only
    process ids and numbers are read; nothing of the counter's path (it names the machine) is kept."""

    _PID = re.compile(r"pid_(\d+)_")

    def __init__(self, loop: Loop, daemon) -> None:
        self.loop, self.daemon = loop, daemon
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        if shutil.which("typeperf"):
            self.thread.start()

    def stop(self) -> None:
        self._stop.set()

    def sample(self) -> dict | None:
        r = subprocess.run(["typeperf", r"\GPU Engine(*)\Utilization Percentage", "-sc", "2"], capture_output=True, text=True,
                           check=False, timeout=60, creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        lines = [x for x in r.stdout.splitlines() if x.startswith('"')]
        if len(lines) < 3:
            return None
        head = next(iter(__import__("csv").reader([lines[0]])))
        vals = next(iter(__import__("csv").reader([lines[-1]])))
        per: dict[int, float] = defaultdict(float)
        for h, v in zip(head[1:], vals[1:]):
            m = self._PID.search(h)
            if not m:
                continue
            try:
                x = float(v)
            except ValueError:
                continue
            per[int(m.group(1))] = max(per[int(m.group(1))], x)
        names = _python_pids()
        parents = process_parents()
        # a venv's python.exe is a launcher that starts the interpreter as its child: a process is ours with its descendants
        srv = descendants(parents, self.daemon.server_pid) | descendants(parents, getattr(self.daemon, "judge_pid", None))
        trn = descendants(parents, self.daemon.trainer_pid)
        inf = sum(per.get(p, 0.0) for p in srv)
        tr = sum(per.get(p, 0.0) for p in trn)
        mine = srv | trn
        lab = sum(x for p, x in per.items() if p in names and p not in mine)
        other = sum(x for p, x in per.items() if p not in names and p not in mine)
        return {"t": stamp(), "inference": round(inf, 1), "training": round(tr, 1), "lab_python": round(lab, 1),
                "other": round(other, 1), "phase": self.daemon.phase}

    def _run(self) -> None:
        while not self._stop.wait(60):
            with contextlib.suppress(Exception):
                s = self.sample()
                if s is not None:
                    with open(self.loop.root / "gpu.jsonl", "a", encoding="utf-8") as fh:
                        fh.write(json.dumps(s) + "\n")


def process_parents() -> dict[int, int]:
    """{pid: parent pid} of every process (a Toolhelp snapshot); {} off Windows."""
    if os.name != "nt":
        return {}
    import ctypes  # noqa: PLC0415
    from ctypes import wintypes  # noqa: PLC0415

    class PE(ctypes.Structure):
        _fields_ = [("dwSize", wintypes.DWORD), ("cntUsage", wintypes.DWORD), ("th32ProcessID", wintypes.DWORD),
                    ("th32DefaultHeapID", ctypes.c_size_t), ("th32ModuleID", wintypes.DWORD), ("cntThreads", wintypes.DWORD),
                    ("th32ParentProcessID", wintypes.DWORD), ("pcPriClassBase", ctypes.c_long), ("dwFlags", wintypes.DWORD),
                    ("szExeFile", ctypes.c_char * 260)]

    k32 = ctypes.windll.kernel32
    k32.CreateToolhelp32Snapshot.restype = wintypes.HANDLE
    snap = k32.CreateToolhelp32Snapshot(0x2, 0)  # TH32CS_SNAPPROCESS
    out: dict[int, int] = {}
    if not snap or snap == wintypes.HANDLE(-1).value:
        return out
    try:
        e = PE()
        e.dwSize = ctypes.sizeof(PE)
        ok = k32.Process32First(snap, ctypes.byref(e))
        while ok:
            out[int(e.th32ProcessID)] = int(e.th32ParentProcessID)
            ok = k32.Process32Next(snap, ctypes.byref(e))
    finally:
        k32.CloseHandle(snap)
    return out


def descendants(parents: dict[int, int], root: int | None) -> set[int]:
    """`root` and every process below it."""
    if not root:
        return set()
    kids: dict[int, list[int]] = defaultdict(list)
    for c, p in parents.items():
        if c != p:
            kids[p].append(c)
    out, stack = set(), [int(root)]
    while stack:
        x = stack.pop()
        if x not in out:
            out.add(x)
            stack.extend(kids.get(x, []))
    return out


def _python_pids() -> set[int]:
    r = subprocess.run(["tasklist", "/FO", "CSV", "/NH"], capture_output=True, text=True, check=False,
                       creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
    out = set()
    for row in __import__("csv").reader(r.stdout.splitlines()):
        if len(row) > 1 and row[0].lower().startswith("python"):
            with contextlib.suppress(ValueError):
                out.add(int(row[1]))
    return out


def gpu_split(loop: Loop, since: str | None = None) -> dict:
    rows = [r for r in fg.read_jsonl(loop.root / "gpu.jsonl") if since is None or r["t"] >= since]
    n = len(rows)
    if not n:
        return {"samples": 0}
    return {"samples": n, **{k: round(sum(r[k] for r in rows) / n, 1) for k in ("inference", "training", "lab_python", "other")}}


# ---------------------------------------------------------------------------------------------------------- the daemon


def generating(a) -> bool:
    """The daemon plays its own practice segments (`--procs` of 1 or more). `--procs 0` is generation off (D65 addendum 4): the rows the
    trainer learns from come from elsewhere (`--trainer-extra-rows-root`), and the daemon only trains, filters and gates."""
    return int(getattr(a, "procs", 1)) > 0


def gate_procs(a) -> int:
    """The processes the gate's games and the human bar's play with: `--gate-procs`, else `--procs` (what they always were)."""
    n = getattr(a, "gate_procs", None)
    return int(a.procs if n is None else n)


def trainer_argv(a, root: Path, inc_dir: Path, parent_pid: int) -> list[str]:
    """The fight trainer's command line. The three `--trainer-*` flags are handed on only when given, so a daemon started without them
    starts the trainer with the line it always had."""
    argv = [GPU_PYTHON, str(ROOT / "rl" / "gym_loop_train.py"), "--root", str(root), "--base", str(inc_dir),
            "--parent-pid", str(parent_pid), "--every-min", str(a.checkpoint_min), "--max-epochs", str(a.max_epochs),
            "--window", str(a.train_window)]
    if getattr(a, "reset_every", 0):
        argv += ["--reset-every", str(a.reset_every)]
    for p in getattr(a, "trainer_extra_rows_root", None) or []:
        argv += ["--extra-rows-root", str(p)]
    if getattr(a, "trainer_extra_window", None):
        argv += ["--extra-window", str(int(a.trainer_extra_window))]
    if getattr(a, "trainer_policy_min_grade", None) is not None:
        argv += ["--policy-min-grade", str(int(a.trainer_policy_min_grade))]
    return argv


class ClockWatch:
    """Every 30 s: past midnight the daily line is written, and the status file's `waiting_s_today` and `playing_s_today` are fresh even in
    the middle of a twenty-minute step (the main thread writes the status only when the phase changes)."""

    def __init__(self, daemon) -> None:
        self.daemon = daemon
        self._stop = threading.Event()
        self.thread = threading.Thread(target=self._run, daemon=True)

    def start(self) -> None:
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()

    def _run(self) -> None:
        while not self._stop.wait(30):
            with contextlib.suppress(Exception):
                self.daemon.clock.roll()
                self.daemon.refresh_status()


class Daemon:
    clock = None  # a `gym_clock.TimeLedger` once `run` starts; a daemon built without one times nothing
    wiring = None  # a `gym_wiring.Wiring` once `run` starts with `--wiring on`; a daemon built without one wires nothing (and never pauses)
    paused: str | None = None  # why the loop is paused: the latest promoted player is used by no run candidate (D68, lane 6c)
    _wiring_t = 0.0
    _status_lock = threading.Lock()

    def __init__(self, a: argparse.Namespace) -> None:
        self.a = a
        self.loop = Loop(a.root)
        self.phase = "start"
        self.server_pid: int | None = None
        self.judge_pid: int | None = None
        self.trainer_pid: int | None = None
        self.trainer = None
        self.job = None
        self.commit = ""
        self.rnd = random.Random(int(time.time()))

    # -- status
    def status(self, phase: str, **kw) -> None:
        self.phase = phase
        self._status_kw = kw
        self.refresh_status()

    def refresh_status(self) -> None:
        """Write the status file: the phase and what it was given, and when a clock runs the day's seconds waited and played (`waiting_s_today`,
        `playing_s_today`, `waits_today` by reason)."""
        doc = {"pid": os.getpid(), "phase": self.phase, "t": stamp(), "root": str(self.loop.root), "commit": self.commit,
               "segment": self.loop.state["segment"], "incumbent": self.loop.state["incumbent"],
               "server_pid": self.server_pid, "trainer_pid": self.trainer_pid, **getattr(self, "_status_kw", {})}
        if getattr(self, "a", None) is not None and hasattr(self.a, "procs"):  # a reader sees at once that generation is off
            doc.update(procs=int(self.a.procs), gate_procs=gate_procs(self.a), generation="on" if generating(self.a) and not self.paused else "off")
        if self.paused:  # D68: generation, gate and trainer stopped until the promoted player is used (or the alert file is removed)
            doc.update(paused=self.paused)
        if self.clock is not None:
            doc.update(self.clock.status_fields())
        with self._status_lock:
            STATUS_DIR.mkdir(parents=True, exist_ok=True)
            try:
                write_json(STATUS_DIR / "status", doc)
            except OSError as error:  # the file is for readers (the lab's runner, the checks): a missed write must not end the daemon
                self.loop.log(f"status: not written ({type(error).__name__}: {error})")

    # -- time: what the daemon waits for and what it plays
    def _span(self, kind: str, what: str, log: bool = False):
        return self.clock.span(kind, what, log=log) if self.clock is not None else contextlib.nullcontext()

    def _block(self, why: str) -> None:
        """Something stops a start: the wait is timed under its reason and logged when it starts (and, in `_unblock`, when it ends)."""
        if self.clock is not None and why != "stop":
            self.clock.wait(why)

    def _unblock(self) -> None:
        if self.clock is not None:
            self.clock.clear()

    def plateau(self, force: bool = False) -> None:
        """D57 addendum 2: no promotion for five days or twenty-five gated candidates writes LAB/alerts/gym-plateau.txt (and a log line);
        the next promotion removes it. Checked at most every ten minutes and after every verdict; off unless `--plateau-alert on`."""
        if getattr(self.a, "plateau_alert", False) not in (True, "on"):
            return
        if not force and time.time() - getattr(self, "_plateau_t", 0.0) < 600:
            return
        self._plateau_t = time.time()
        try:
            gg.plateau_update(gg.read_ledger(self.loop.gates / "verdicts.jsonl"), self.loop.state["promotions"], self.loop.state["incumbent"],
                              Path(getattr(self.a, "alerts_dir", None) or LAB / "alerts"), self.loop.log, utc())
        except Exception as error:  # noqa: BLE001 -- an alarm that cannot be written must not end the loop
            if not getattr(self, "_plateau_err", False):
                self._plateau_err = True
                self.loop.log(f"plateau: not written ({type(error).__name__}: {error})")

    # -- the fight brain's way into the run player (D68, lane 6c; rl/gym_wiring.py)
    def make_wiring(self):
        """`--wiring on` (the default of `run`): the lab's folders and the builder. A daemon built without the option wires nothing."""
        a = self.a
        if getattr(a, "wiring", "off") != "on":
            return None
        return gw.Wiring(agent_root=Path(a.agent_root), train_dir=Path(a.train_dir), alerts_dir=Path(a.alerts_dir), jobs_json=Path(a.jobs_json),
                         partial_dir=self.loop.root / "partial-builds", unconsumed_h=float(a.unconsumed_h), builder=self._build_in_span)

    def _build_in_span(self, *args) -> Path:
        """The builder (a child process in the GPU venv, a few minutes): timed as a play step `wiring`, so the clock shows it."""
        self.status("wiring", why="building the run candidate of the promoted fight brain")
        with self._span("play", "wiring"):
            return gw.subprocess_builder(*args)

    def wiring_tick(self, queue: bool = False, force: bool = False) -> None:
        """One look at the latest promoted player: joined to the run judge's verdict when it has one, queued as a run candidate when no
        candidate carries it (at most every ten minutes, and only when `queue`: a step may start now; always after a promotion, `force`), and
        the pause decided (`self.paused`). A failure of the look is logged once and never ends the loop."""
        w = self.wiring
        if w is None:
            return
        do_queue = (queue and time.time() - self._wiring_t >= 600) or force
        if do_queue:
            self._wiring_t = time.time()
        before = json.dumps(self.loop.state.get("wiring"), sort_keys=True, default=str)
        try:
            res = gw.evaluate(w, self.loop.state, dt.datetime.now(dt.timezone.utc), log=self.loop.log, gates_dir=self.loop.gates, queue=do_queue)
        except Exception as error:  # noqa: BLE001 -- the wiring is a side job: a failure is logged once and the loop goes on
            if not getattr(self, "_wiring_err", False):
                self._wiring_err = True
                self.loop.log(f"wiring: {type(error).__name__}: {error}")
            return
        self.paused = res.get("paused")
        if json.dumps(self.loop.state.get("wiring"), sort_keys=True, default=str) != before:
            self.loop.save()

    def pause_unconsumed(self) -> None:
        """The latest promoted player is used by no run candidate 24 hours after its promotion: generation, gate and trainer stop, timed as the
        wait `unconsumed` (logged when it starts and ends like every wait); the next look in 30 s sees the player consumed or the alert removed."""
        self._block("unconsumed")
        hold_quiet(False)
        self.status("paused", why=self.paused)
        self.stop_trainer()
        time.sleep(30)

    # -- the trainer process
    def start_trainer(self) -> None:
        if self.a.no_trainer or (self.trainer is not None and self.trainer.poll() is None):
            return
        if self.trainer is not None:  # it ended on its own: logged, and started again at most every ten minutes
            self.loop.log(f"trainer: ended rc={self.trainer.returncode}")
            self.trainer = None
            self.trainer_pid = None
        if time.time() - getattr(self, "_trainer_t", 0.0) < 600:
            return
        self._trainer_t = time.time()
        from tools.lab import winsys  # noqa: PLC0415

        inc, inc_dir = self.loop.incumbent()
        argv = trainer_argv(self.a, self.loop.root, inc_dir, os.getpid())
        logf = open(self.loop.trainer / "trainer.out", "a", encoding="utf-8")  # noqa: SIM115 -- the child's for its life
        self.trainer = winsys.suspended_popen(argv, self.job, stdout=logf, stderr=subprocess.STDOUT, cwd=str(ROOT),
                                              creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
        self.trainer_pid = self.trainer.pid
        self.loop.log(f"trainer: started (pid {self.trainer_pid})")

    def stop_trainer(self) -> None:
        if self.trainer is None:
            return
        (self.loop.trainer / "STOP").write_text("stop\n", encoding="utf-8")
        try:
            self.trainer.wait(120)
        except subprocess.TimeoutExpired:
            self.trainer.kill()  # our own child, by its handle
        (self.loop.trainer / "STOP").unlink(missing_ok=True)
        self.trainer = None
        self.trainer_pid = None

    # -- the steps
    def next_gate(self) -> tuple[str, Path] | None:
        first = getattr(self.a, "first_candidate", "") or ""
        if first and active_rule(self.a) is PROMOTION_RULE_V4:
            # D57 addendum 9: the checkpoint the rule's first gate is on (c0380, whose exam games exist), until it has a verdict
            d = self.loop.checkpoints / first
            v = self.loop.state["gated"].get(first)
            if d.is_dir() and (d / "combat_policy.pt").is_file() and not (isinstance(v, dict) and v.get("verdict") in ("reject", "promote")):
                return first, d
        if self.a.prefilter == "off":
            newest = self.loop.newest_checkpoint()
            if newest is None or newest[0] in self.loop.state["gated"]:
                return None
            every = float(getattr(self.a, "gate_every_min", 0.0) or 0.0)
            if every > 0 and not gate_offered(newest[1], gate_anchor(self.loop), every):
                return None
            return newest
        return prefilter_pick(self.loop, prefilter_cfg(self.a))

    def gate(self, cand: str, cand_dir: Path) -> None:
        v5 = active_rule(self.a) is PROMOTION_RULE_V5
        v4 = active_rule(self.a) is PROMOTION_RULE_V4
        self.status("gate", candidate=cand)
        with self._span("play", "gate"):
            if v5:
                v = run_gate_v5(self.loop, cand, cand_dir, gate_procs(self.a), self.a.gate_k, daemon=self)
            elif v4:
                v = run_gate_v4(self.loop, cand, cand_dir, gate_procs(self.a), self.a.gate_k, daemon=self)
            else:
                v = run_gate(self.loop, cand, cand_dir, self.a.player_config, gate_procs(self.a), self.a.gate_k, self.a.prior, daemon=self,
                             judge=self.a.judge or None)
        if v.get("verdict") == "stopped":
            return
        if v.get("stage") in ((2, 3) if v4 or v5 else (2,)):
            v = promotion_guard(self.loop, v, active_rule(self.a))
            if v.get("guard"):
                self.loop.log(f"gate {cand}: {v['why']}")
        mode = getattr(self.a, "human_bar", "promotion")
        if v4 or v5:
            pass  # the human line is part of the v4 and v5 verdicts (stages 2 and 3, from the exam games the verdict stands on); `--human-bar` is a no-op
        elif v.get("stage") == 2 and ((mode == "on" and v.get("verdict") in ("promote", "reject")) or (mode == "promotion" and v.get("verdict") == "promote")):
            # the human line (D55 addendum 11): report-only, after the rule has spoken; a stage-1 reject never plays it, and in mode
            # `promotion` (D57 item 2) only a promotion does, once, on the CPU prior
            self.status("gate", candidate=cand, part="human bar")
            try:
                with self._span("play", "human-bar"):
                    v["human"] = human_bar(self.loop, cand, cand_dir, v["incumbent"], gate_procs(self.a), self.a.gate_k, HUMAN_BAR["prior"],
                                           daemon=self)
            except Exception as error:  # noqa: BLE001 -- a reading that fails must never change or lose the verdict
                v["human"] = f"error: {type(error).__name__}"
                self.loop.log(f"gate {cand}: human bar: {type(error).__name__}: {error}")
        if self.a.prefilter != "off":
            prefilter_note_verdict(self.loop, cand, v)
        older = sorted(p.name for p in self.loop.checkpoints.glob("c[0-9][0-9][0-9][0-9]") if p.name < cand)
        for old in older:  # superseded while untested: the newest is always the one scored
            self.loop.state["gated"].setdefault(old, {"verdict": "superseded", "by": cand})
        self.loop.state["gated"][cand] = {k: v.get(k) for k in ("verdict", "stage", "why", "pairs", "cand_wins", "inc_wins",
                                                                "win_diff", "win_se", "l0_diff", "p", "incumbent", "config",
                                                                "stage1_js", "human", "path", "diff_lb", "pool_k", "pool_diff", "pool_p",
                                                                "guard", "rule", "draws", "hp_diff", "act3", "era") if k not in (
                                                                    "human", "path", "diff_lb", "pool_k", "pool_diff", "pool_p", "guard",
                                                                    "rule", "draws", "hp_diff", "act3", "era") or k in v}
        self.loop.save()
        with open(self.loop.gates / "verdicts.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": stamp(), **{k: v.get(k) for k in v if k not in ("cand_file", "inc_file")}}) + "\n")
        if v.get("verdict") == "promote":
            ver = promote(self.loop, cand, cand_dir, v, self.a.board)
            if "human" in v and not v5:  # a v5 incumbent plays its own era's draws; nothing is adopted
                try:
                    human_bar_adopt(self.loop, cand, ver)
                except OSError as error:
                    self.loop.log(f"gate {cand}: human bar: games not kept as {ver}'s ({type(error).__name__}: {error})")
        self.plateau(force=True)  # the alarm follows the verdict: a promotion takes it down
        if v.get("verdict") == "promote":
            self.wiring_tick(queue=True, force=True)  # D68: the promoted player's run candidate is built and queued at once

    def teacher_step(self) -> bool:
        """D60 item 5: the run teacher's elite and boss fights as a fourth source of the practice mix. The set follows the teacher's store the
        way the nights' follows the nights: when it holds none yet, or `--teacher-min-new` finished runs have come since, its new runs are
        replayed (`rl/teacher_starts.py`; only those) and the set is assembled again. Off unless `--teacher-store` names the store. A build
        that fails is logged and left alone for an hour. True when it built (the caller waits its turn and goes round again)."""
        store = getattr(self.a, "teacher_store", "")
        if not store or time.time() < getattr(self, "_teacher_hold", 0.0):
            return False
        import teacher_starts as tst  # noqa: PLC0415

        gens = tuple(int(x) for x in str(getattr(self.a, "teacher_gens", "14,15")).split(",") if x.strip())
        try:
            why = tst.stale(self.loop, Path(store), gens, int(getattr(self.a, "teacher_min_new", tst.MIN_NEW)))
        except Exception as error:  # noqa: BLE001 -- an unreadable store must not end the loop
            self.loop.log(f"teacher starts: store unreadable ({type(error).__name__}: {error})")
            self._teacher_hold = time.time() + 3600
            return False
        if not why:
            return False
        self.status("starts", why=why)
        self.loop.log(f"teacher starts: building ({why})")
        try:
            with self._span("play", "teacher-starts"):
                res = tst.build(self.loop, Path(store), gens, procs=self.a.procs)
        except Exception as error:  # noqa: BLE001 -- logged; the state on disk stays, the next try is an hour on
            self.loop.log(f"teacher starts: build failed ({type(error).__name__}: {error})")
            self._teacher_hold = time.time() + 3600
            return False
        if isinstance(res, dict) and res.get("stopped"):
            self.loop.log(f"teacher starts: stopped ({res['stopped']})")
        return True

    def segment(self) -> None:
        with self._span("play", "generate"):
            self._segment()

    def idle(self) -> None:
        """Generation is off (`--procs 0`) and no checkpoint is offered to the gate: the trainer is the only worker. Keep the newest
        checkpoints (the step after a segment does it when generating), then wait `--idle-s` for the next one: timed as the wait `idle`
        (counted in `waits_today`, not logged line by line, as the yield), cut short by the loop's STOP."""
        hold_quiet(False)
        prune_checkpoints(self.loop)
        self.status("idle", why="generation off: no checkpoint offered to the gate")
        if self.clock is not None:
            self.clock.wait("idle", log=False)
        end = time.time() + float(getattr(self.a, "idle_s", 60.0))
        while time.time() < end and not (self.loop.root / "STOP").exists():
            time.sleep(5)
        self._unblock()

    def run_record(self) -> dict:
        """What this run of the daemon was started with that a reader of `state.json` needs: who generates, who gates, what the trainer reads."""
        a = self.a
        return {"t": stamp(), "generation": "on" if generating(a) else "off", "procs": int(a.procs), "gate_procs": gate_procs(a),
                "trainer": {"extra_rows_root": [str(p) for p in (getattr(a, "trainer_extra_rows_root", None) or [])],
                            "extra_window": getattr(a, "trainer_extra_window", None),
                            "policy_min_grade": getattr(a, "trainer_policy_min_grade", None),
                            "window": a.train_window, "max_epochs": a.max_epochs, "checkpoint_min": a.checkpoint_min}}

    def run_line(self) -> str:
        """The start line's account of the processes and of what the trainer reads."""
        a, rec = self.a, self.run_record()["trainer"]
        who = (f"generation {a.procs} processes K={a.k}, gate {gate_procs(a)} processes" if generating(a)
               else f"generation off (--procs 0), gate {gate_procs(a)} processes")
        reads = (f"the plain rows of {', '.join(rec['extra_rows_root'])} (extra window {rec['extra_window'] or 'none'}, policy from grade "
                 f"{'16' if rec['policy_min_grade'] is None else rec['policy_min_grade']})" if rec["extra_rows_root"] else "its own shards only")
        return f"{who}; the trainer reads {reads}"

    def _segment(self) -> None:
        loop = self.loop
        loop.state["segment"] += 1
        seg = loop.state["segment"]
        player, pdir = loop.generation_player()
        tasks = sample_tasks(loop, player, self.a.procs * self.a.segment_min * 60, self.a.teacher, self.a.exact_wall_s, self.rnd,
                             preloss=not getattr(self.a, "no_preloss", False))
        loop.save()
        target = loop.games / f"seg{seg:06d}.jsonl"
        self.status("generate", player=player, tasks=len(tasks))
        loop.log(f"segment {seg}: {len(tasks)} fights by {player} under {self.a.teacher}; {pmix.mix_line(loop.mix)}")
        t0 = time.time()
        stopped = play_tasks(loop, tasks, target, pdir, self.a.procs, self.a.k, self.a.prior, f"seg{seg}", daemon=self)
        rows = fg.read_jsonl(target)
        learn_seconds(loop, rows)
        entry = compact(loop, seg, value_labels=stopped is None)
        pre: dict = {}
        if not getattr(self.a, "no_preloss", False):
            pre = pmix.merge_preloss(loop.root, int(pmix.config(loop.starts)["preloss"]["max"]))
            if pre["merged"] or pre["dropped"]:
                loop.log(f"preloss: {pre['merged']} new starts from lost fights, {pre['kept']} kept, {pre['dropped']} dropped")
        loop.save()
        with open(loop.games / "segments.jsonl", "a", encoding="utf-8") as fh:
            fh.write(json.dumps({"t": stamp(), "segment": seg, "player": player, "fights": len(rows), "tasks": len(tasks),
                                 "wall_s": round(time.time() - t0, 1), "stopped": stopped,
                                 "capped": sum(1 for r in rows if r.get("capped")), "samples": entry["rows"] if entry else 0,
                                 "cats": dict(Counter(r["cat"] for r in rows)),
                                 "won": dict(Counter(r["cat"] for r in rows if r.get("won"))),
                                 "mix": loop.mix.get("by_kind"), "regret_quartiles": loop.mix.get("regret_quartiles"),
                                 "preloss": dict(Counter(r["preloss"] for r in rows if r.get("preloss"))),
                                 **({"preloss_set": pre} if pre else {})}) + "\n")
        retain(loop, self.a.buffer_rows, self.a.buffer_days, self.a.buffer_gb)
        prune_checkpoints(loop)

    def wait_free(self) -> None:
        """Drop the line and let a waiting step in: wait `yield_s`, then while anything blocks a start."""
        hold_quiet(False)
        self.status("yield")
        if self.clock is not None:
            self.clock.wait("yield", log=False)  # a fixed pause by design: timed, not logged line by line
        end = time.time() + self.a.yield_s
        while time.time() < end and not (self.loop.root / "STOP").exists():
            time.sleep(5)
        why = why_blocked(self.loop.root, starting=True)
        while why and why != "stop":
            self._block(why)
            self.status("waiting", why=why)
            time.sleep(30)
            why = why_blocked(self.loop.root, starting=True)
        self._unblock()

    def run(self) -> int:
        from tools.lab import winsys  # noqa: PLC0415

        winsys.set_below_normal()
        if not winsys.acquire_mutex("spireline-gym-loop"):
            print("another gym loop is running")
            return 2
        self.job = winsys.JobObject(kill_on_close=True)  # everything started from here dies with the daemon, however it ends
        self.job.assign_self()
        self.commit = _git("rev-parse", "--short", "HEAD", cwd=ROOT).strip()
        (self.loop.root / "STOP").unlink(missing_ok=True)
        self.loop.log(f"daemon: start pid {os.getpid()} at {self.commit} player config {self.a.player_config}, teacher "
                      f"{self.a.teacher}, {self.run_line()}")
        self.loop.state["run"] = self.run_record()  # what this run was started with: who generates, who gates, what the trainer reads
        self.loop.save()
        for root in getattr(self.a, "trainer_extra_rows_root", None) or []:
            if not Path(root).is_dir():
                self.loop.log(f"daemon: the trainer's plain-rows root {root} is not there (yet): it reads what appears")
        rule = active_rule(self.a)
        self.loop.log(f"daemon: state reloaded: incumbent {self.loop.state['incumbent']}, {len(self.loop.state['gated'])} checkpoints "
                      f"with a verdict, segment {self.loop.state['segment']}; {rule['name']}: "
                      + (v5_rule_line(self.loop, rule) if rule is PROMOTION_RULE_V5 else "")
                      + (f"the exam's human stratum under {rule['config']} ({rule['prior']} prior), stages of {list(rule['stage_draws'])} draws; "
                         f"promote at {rule['promote']['points']:+g} points with the 90% lower bound above 0 and HP lost not worse by more than "
                         f"{rule['promote']['hp_worse']:g}; Act 3 guard from {rule['act3_guard']['after_h']:g} h after the first Act 3 practice segment"
                         + (f"; first candidate {self.a.first_candidate}" if getattr(self.a, "first_candidate", "") else "")
                         if rule is PROMOTION_RULE_V4 else "" if rule is PROMOTION_RULE_V5
                         else f"stage 1 plays j {list(stage1_js(PROMOTION_RULE, self.loop.gate_pair()))}"))
        pcfg = prefilter_cfg(self.a)
        self.loop.log(f"daemon: gate pre-filter {PREFILTER['name']}: mode {pcfg['mode']}, margin {pcfg['margin']} nats, floor "
                      f"{pcfg['floor_h']} h, on {PREFILTER['metric']}")
        if getattr(self.a, "gate_every_min", 0.0):
            self.loop.log(f"daemon: gate cadence: a checkpoint is offered to the gate {self.a.gate_every_min:g} minutes after the last one gated "
                          f"(the trainer writes one every {self.a.checkpoint_min:g} minutes, passes capped at {self.a.max_epochs:g} a row)")
        mode = getattr(self.a, "human_bar", "promotion")
        if rule is PROMOTION_RULE_V5:
            self.loop.log("daemon: human bar: under rule v5 every stage-2 or stage-3 verdict carries the human line from its own ruler's human starts "
                          f"(report-only; a new series beside the exam's); `--human-bar {mode}` is a no-op")
        elif rule is PROMOTION_RULE_V4:
            self.loop.log("daemon: human bar: under rule v4 every stage-2 or stage-3 verdict carries the human line from its own exam games "
                          f"(report-only); `--human-bar {mode}` is a no-op")
        else:
            self.loop.log(f"daemon: human bar {mode}: " + {"promotion": "a promotion", "on": "every stage-2 verdict", "off": "nothing"}.get(mode, mode)
                          + f" is followed by the human stratum ({HUMAN_BAR['draws']} draws under {HUMAN_BAR['config']}, {HUMAN_BAR['prior']} prior), report-only")
        self.loop.log(f"daemon: plateau alarm {'on' if getattr(self.a, 'plateau_alert', False) in (True, 'on') else 'off'}: {gg.PLATEAU['days']:g} days or "
                      f"{gg.PLATEAU['candidates']} gated candidates without a promotion writes {gg.PLATEAU['file']}")
        self.wiring = self.make_wiring()
        self.loop.log("daemon: wiring " + (f"on: a promoted player is built into the champion as agent/gen<k+1>/candidate-fight-v<N> and queued by a marker in "
                                           f"{self.wiring.train_dir.name}; a player in no champion and no queued or judged candidate {self.wiring.unconsumed_h:g} h after its "
                                           f"promotion pauses the loop and writes {gw.ALERT_FILE}" if self.wiring else "off: a promoted player is not built into a run candidate"))
        if generating(self.a):
            mcfg = pmix.config(self.loop.starts)
            self.loop.log(f"daemon: practice mix: human share {json.dumps(mcfg['human_share'])}, regret {json.dumps(mcfg['regret'])}, "
                          f"pre-loss starts {'off' if getattr(self.a, 'no_preloss', False) else 'on, ' + json.dumps(mcfg['preloss'])}")
            if getattr(self.a, "teacher_store", ""):
                self.loop.log(f"daemon: teacher starts: the store's generations {self.a.teacher_gens} are replayed into starts/teacher-starts "
                              f"(a rebuild from {self.a.teacher_min_new} new runs), teacher share {json.dumps(mcfg['teacher_share'])}")
        else:
            self.loop.log("daemon: generation off: no segment is played, no start set, weights or retention step runs; the trainer, the "
                          "pre-filter and the gate do (the loop's own shards stay as the older rows)")
            if getattr(self.a, "teacher_store", ""):
                self.loop.log(f"daemon: teacher starts: skipped (generation off; {self.a.teacher_store} feeds only the practice's start mix)")
        self.clock = gclock.TimeLedger(self.loop.games / "clock.jsonl", self.loop.log, segments=self.loop.games / "segments.jsonl",
                                       ledger=self.loop.gates / "verdicts.jsonl", run_judge=self.loop.gates / "run-judge.jsonl")
        self.loop.clock = self.clock
        self.clock.roll()  # a day finished while the daemon was down gets its daily line now
        cwatch = ClockWatch(self)
        cwatch.start()
        watch = GpuWatch(self.loop, self)
        watch.start()
        self.status("start")
        self.plateau(force=True)
        self.wiring_tick(queue=why_blocked(self.loop.root, starting=True) is None)  # the books are joined and a waiting player queued at a start
        entry = compact(self.loop, self.loop.state["segment"], value_labels=False)  # a killed segment's finished fights
        if entry:
            self.loop.log(f"buffer: {entry['fights']} fights of an interrupted segment kept")
        rc = 0
        try:
            while not (self.loop.root / "STOP").exists():
                if self.a.until and stamp() >= self.a.until:
                    self.loop.log("daemon: --until reached")
                    break
                self.plateau()
                why = why_blocked(self.loop.root, starting=True)
                self.wiring_tick(queue=not why)  # D68: joins the run judge's word, queues the promoted player's candidate (only when a step may start)
                if self.paused:
                    self.pause_unconsumed()  # nothing spends the card on a fight brain no run uses; resumes when the player is consumed
                    continue
                if why:
                    self._block(why)
                    self.status("waiting", why=why)
                    self.start_trainer()  # the night is training time: the trainer keeps its own rules (low CPU, GPU_OK)
                    time.sleep(30)
                    continue
                self._unblock()
                generate = generating(self.a)  # --procs 0: the start set, the weights and the segments feed only the generation
                if generate:
                    nights = nights_available(max_nights=self.a.max_nights)
                    stale = starts_stale(self.loop, nights, self.a.runs_per_night)
                    if stale:  # the set follows the nights: a finished night's runs join it (only theirs are replayed)
                        self.status("starts", why=stale)
                        self.loop.log(f"starts: rebuilding ({stale})")
                        with self._span("play", "starts"):
                            build_starts(self.loop, nights, runs_per_night=self.a.runs_per_night, procs=self.a.procs)
                        self.wait_free()
                        continue
                    if self.teacher_step():  # the teacher's fights follow the teacher's store (D60 item 5; off without --teacher-store)
                        self.wait_free()
                        continue
                    stale = weights_stale(self.loop)
                    if stale:
                        self.status("weights", why=stale)
                        self.loop.log(f"weights: computing ({stale})")
                        with self._span("play", "weights"):
                            kind_weights(self.loop, procs=self.a.procs, prior=self.a.prior)
                        self.wait_free()
                        continue
                self.start_trainer()
                with self._span("play", "prefilter"):  # who enters the gate (it can score a checkpoint on the CPU for minutes)
                    cand = self.next_gate()
                if cand is not None and not self.a.no_gate:
                    self.gate(*cand)
                elif generate:
                    self.segment()
                else:
                    self.idle()  # nothing to gate and nothing to generate: wait for the trainer's next checkpoint
                    continue
                self.wait_free()
        except Exception:  # noqa: BLE001 -- logged; the state on disk stays for the restart
            import traceback  # noqa: PLC0415

            self.loop.log("daemon: " + traceback.format_exc()[-1500:])
            rc = 1
        finally:
            hold_quiet(False)
            drop_gpu(TAG)
            self.stop_trainer()
            watch.stop()
            cwatch.stop()
            self.clock.end()
            self.loop.save()
            self.status("stopped", rc=rc)
            self.loop.log(f"daemon: stopped rc={rc}")
        return rc


# ---------------------------------------------------------------------------------------------------------- commands


def cmd_stop(a) -> int:
    """Ask the daemon to stop (its STOP file); after `--wait-s`, end it by the PID its status names (its Job Object takes its
    children with it)."""
    loop = Loop(a.root)
    (loop.root / "STOP").write_text("stop\n", encoding="utf-8")
    st = read_json(STATUS_DIR / "status", {}) or {}
    pid = st.get("pid")
    end = time.time() + a.wait_s
    from tools.lab import winsys  # noqa: PLC0415

    while pid and time.time() < end and winsys.parent_alive(int(pid)):
        time.sleep(5)
    if pid and winsys.parent_alive(int(pid)):
        subprocess.run(["taskkill", "/PID", str(pid), "/F"], capture_output=True, check=False)
        print(f"killed {pid}")
    else:
        print("stopped")
    return 0


def cmd_status(a) -> int:
    loop = Loop(a.root)
    st = read_json(STATUS_DIR / "status", {})
    seg = fg.read_jsonl(loop.games / "segments.jsonl")
    idx = fg.read_jsonl(loop.root / "buffer" / "index.jsonl")
    print(json.dumps({"status": st, "segments": len(seg), "fights": sum(s["fights"] for s in seg),
                      "samples": sum(e["rows"] for e in idx), "checkpoints": len(list(loop.checkpoints.glob("c*"))),
                      "incumbent": loop.state["incumbent"], "gated": len(loop.state["gated"]), "gpu": gpu_split(loop)}, indent=1))
    return 0


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("starts", "weights", "run", "gate", "stop", "status", "import-games", "gate5-set"):
        p = sub.add_parser(name)
        p.add_argument("--root", default=str(DEFAULT_ROOT))
        p.add_argument("--procs", type=int, default=6,
                       help="processes the generation plays with (and the start set's and weights' builds); `run --procs 0` is generation "
                            "off: no segment, start set, weights or teacher-starts step, only the trainer, the pre-filter and the gate "
                            "(then `--gate-procs` is needed)")
        p.add_argument("--prior", choices=("server", "cpu"), default="server")
        if name in ("starts", "run"):
            p.add_argument("--runs-per-night", type=int, default=RUNS_PER_NIGHT)
            p.add_argument("--max-nights", type=int, default=MAX_NIGHTS, help="the newest finished nights the set takes (0: all)")
        if name in ("run", "gate"):
            p.add_argument("--gate-procs", type=int, default=None,
                           help="processes the gate's games and the human bar play with (default: --procs); lets a gate play with 4 "
                                "while the generation has none")
            p.add_argument("--player-config", default="judge-v0",
                           help="the search config the gate scores: the judge's (D41), a gym arm or a search_configs name")
            p.add_argument("--gate-k", type=int, default=32)
            p.add_argument("--rule", choices=("v5", "v4", "v3"), default="v5",
                           help="the promotion rule: v5 (D68 addendum 5 decision 4, the default), its own ruler of Acts 1-2 human starts outside "
                                "the exam and the Act 3 bosses, two routes, fresh draws per era (`gate5-set` builds the ruler); v4 (D57 addendum 9), "
                                "the fight exam's human stratum; v3, the board's 200 Act 1 boss starts under the judge-equipped search")
            p.add_argument("--judge", default=str(REF / "agent" / "judge" / "v0" / "judge.pt"),
                           help="the judge checkpoint a judge config's leaves read ('' for a config without one)")
        if name == "gate":
            p.add_argument("--candidate", required=True)
        if name == "import-games":
            p.add_argument("--into", required=True, help="the player whose exam games file (exam/games-<into>.jsonl) gets the rows")
            p.add_argument("--games", required=True, help="the games file to take rows from (the rows the file lacks, by start and draw)")
            p.add_argument("--provenance", required=True, help="written into every added row")
        if name == "stop":
            p.add_argument("--wait-s", type=float, default=300.0)
        if name == "gate5-set":
            p.add_argument("--seed", type=int, default=None, help="the draw's seed (the rule's `set.seed` by default)")
            p.add_argument("--rebuild", action="store_true", help="replace an existing gate5/ (its files get a dated backup name)")
        if name == "run":
            p.add_argument("--teacher", default="x16", help="the generation's search: the stronger search grades the fights")
            p.add_argument("--k", type=int, default=2, help="fights a lockstep process holds during generation (few: a stop loses them)")
            p.add_argument("--segment-min", type=float, default=20.0)
            p.add_argument("--yield-s", type=float, default=150.0)
            p.add_argument("--quiet-allow", default=",".join(QUIET_ALLOW),
                           help="QUIET.lock tags the daemon plays beside, comma-separated (a lane's measurement); lab-train still holds it")
            p.add_argument("--exact-wall-s", type=float, default=0.25)
            p.add_argument("--checkpoint-min", type=float, default=30.0)
            p.add_argument("--gate-every-min", type=float, default=0.0,
                           help="D60 item 5: offer the gate a checkpoint only this many minutes after the checkpoint it last played "
                                "(the trainer still writes one every --checkpoint-min; 120 offers one in four); 0: every checkpoint, as before")
            p.add_argument("--max-epochs", type=float, default=4.0, help="the trainer's pass cap over each new row")
            p.add_argument("--reset-every", type=int, default=0,
                           help="the trainer re-initialises the heads' last layers after every N-th checkpoint (0: never)")
            p.add_argument("--train-window", type=int, default=300_000)
            p.add_argument("--idle-s", type=float, default=60.0,
                           help="with --procs 0 and no checkpoint to gate: seconds between two looks for one (timed as the wait `idle`)")
            p.add_argument("--trainer-extra-rows-root", action="append", default=[], metavar="PATH",
                           help="D65 addendum 4: a plain-runs store (or one generation folder of it) whose fight rows the trainer reads "
                                "beside its own shards (gym_loop_train.py --extra-rows-root; repeatable)")
            p.add_argument("--trainer-extra-window", type=int, default=None, metavar="N",
                           help="at most N rows from the --trainer-extra-rows-root parts (the trainer's --extra-window; not given: no cap "
                                "of their own)")
            p.add_argument("--trainer-policy-min-grade", type=int, default=None, metavar="N",
                           help="a plain row's policy is learned from at this grade (stakes multiplier) or deeper (the trainer's "
                                "--policy-min-grade; not given: its default, 16)")
            p.add_argument("--buffer-rows", type=int, default=3_000_000)
            p.add_argument("--buffer-days", type=float, default=21.0)
            p.add_argument("--buffer-gb", type=float, default=20.0)
            p.add_argument("--board", choices=("commit", "dry", "off"), default="commit")
            p.add_argument("--until", default="", help="stop at this UTC stamp (the dry run)")
            p.add_argument("--no-trainer", action="store_true")
            p.add_argument("--no-gate", action="store_true")
            p.add_argument("--first-candidate", default="",
                           help="under rule v4: the checkpoint the first gate plays (its exam games imported with `import-games`), until it has a verdict")
            p.add_argument("--human-bar", choices=("promotion", "on", "off"), default="promotion",
                           help="play the candidate on the fight exam's human stratum (CPU prior) and record the human line "
                                "(report-only; the rule never reads it): promotion, after a stage-2 promotion only (the default); "
                                "on, after every stage-2 verdict; off, never (a no-op under --rule v4, whose verdicts carry the line)")
            p.add_argument("--plateau-alert", choices=("on", "off"), default="on",
                           help="write LAB/alerts/gym-plateau.txt after five days or twenty-five gated candidates without a promotion "
                                "(D57 addendum 2) and remove it at the next promotion")
            p.add_argument("--alerts-dir", default=str(LAB / "alerts"), help="where the plateau alarm's and the unconsumed-player alert's files go")
            p.add_argument("--wiring", choices=("on", "off"), default="on",
                           help="D68 (lane 6c): a promoted player is built into the champion as a run candidate and queued for the lab's judge; "
                                "one that no champion and no queued or judged candidate uses 24 hours after its promotion pauses the loop "
                                "(LAB/alerts/gym-unconsumed.txt; remove the file to resume)")
            p.add_argument("--agent-root", default=str(REF / "agent"), help="the wiring's: the champion (current.json) and the candidates' generation folders")
            p.add_argument("--train-dir", default=str(LAB / "train"), help="the wiring's: the lab's ready-/done- markers")
            p.add_argument("--jobs-json", default=str(LAB / "jobs.json"), help="the wiring's: the lab's config, read for the keep rule its train job judges by")
            p.add_argument("--unconsumed-h", type=float, default=gw.UNCONSUMED_H,
                           help="hours after a promotion that a player used by no run candidate pauses the loop")
            p.add_argument("--teacher-store", default="",
                           help="D60 item 5: the run teacher's store (its <generation>/runs.jsonl, shards and decisions.d); its elite and boss fights "
                                "become the `teacher` source of the start mix, built and followed between steps; '' (the default): off")
            p.add_argument("--teacher-gens", default="14,15", help="the store's generations the teacher starts are made from")
            p.add_argument("--teacher-min-new", type=int, default=25,
                           help="finished teacher runs that make the set stale (the first build takes any number)")
            p.add_argument("--no-preloss", action="store_true",
                           help="do not keep the position three decisions before a lost practice fight as a start")
            p.add_argument("--prefilter", choices=("best", "incumbent", "off"), default=PREFILTER["mode"],
                           help="who enters the gate (D49): best, a checkpoint whose held-out policy CE is below the incumbent's "
                                "and below every rejected checkpoint's; incumbent, below the incumbent's only; off, every one")
            p.add_argument("--prefilter-margin", type=float, default=PREFILTER["margin"],
                           help="nats a checkpoint's held-out policy CE must be below the incumbent's")
            p.add_argument("--prefilter-floor-h", type=float, default=PREFILTER["floor_h"],
                           help="hours after which the best skipped checkpoint is gated whatever its figures")
    return ap


def check_args(ap: argparse.ArgumentParser, a: argparse.Namespace) -> None:
    """The combinations that cannot work: a pool of no processes where one is needed, a trainer option with nothing to apply to."""
    if a.procs < 0:
        ap.error("--procs must be 0 or more")
    if a.cmd in ("starts", "weights") and a.procs < 1:
        ap.error(f"--procs must be 1 or more for `{a.cmd}` (it plays games)")
    if a.cmd in ("run", "gate") and gate_procs(a) < 1 and not (a.cmd == "run" and a.no_gate):
        ap.error("--procs 0 (generation off) needs --gate-procs N of 1 or more: the gate plays with them (or --no-gate)")
    if a.cmd == "run" and not a.trainer_extra_rows_root and (a.trainer_extra_window or a.trainer_policy_min_grade is not None):
        ap.error("--trainer-extra-window and --trainer-policy-min-grade apply to the rows of --trainer-extra-rows-root, which is not given")


def main(argv=None) -> int:
    global QUIET_ALLOW
    ap = build_parser()
    a = ap.parse_args(argv)
    check_args(ap, a)
    if a.cmd == "starts":
        loop = Loop(a.root)
        meta = build_starts(loop, runs_per_night=a.runs_per_night, procs=a.procs, max_nights=a.max_nights)
        print(json.dumps(meta, indent=1)[:2000])
        return 0
    if a.cmd == "weights":
        print(json.dumps(kind_weights(Loop(a.root), procs=a.procs, prior=a.prior), indent=1))
        return 0
    if a.cmd == "gate":
        loop = Loop(a.root)
        d = Path(a.candidate)
        if active_rule(a) is PROMOTION_RULE_V5:
            print(json.dumps(run_gate_v5(loop, d.name, d, gate_procs(a), a.gate_k), indent=1))
        elif active_rule(a) is PROMOTION_RULE_V4:
            print(json.dumps(run_gate_v4(loop, d.name, d, gate_procs(a), a.gate_k), indent=1))
        else:
            print(json.dumps(run_gate(loop, d.name, d, a.player_config, gate_procs(a), a.gate_k, a.prior, judge=a.judge or None), indent=1))
        return 0
    if a.cmd == "import-games":
        print(import_games(a.root, a.into, Path(a.games), a.provenance), "rows added")
        return 0
    if a.cmd == "gate5-set":
        meta = build_gate5(Path(a.root), seed=a.seed, rebuild=a.rebuild)
        print(json.dumps({k: meta[k] for k in ("rule", "written", "seed", "starts", "by_part", "short_cells", "exclusion")}, indent=1))
        return 0
    if a.cmd == "stop":
        return cmd_stop(a)
    if a.cmd == "status":
        return cmd_status(a)
    QUIET_ALLOW = tuple(t.strip() for t in a.quiet_allow.split(",") if t.strip())
    return Daemon(a).run()


if __name__ == "__main__":
    raise SystemExit(main())
