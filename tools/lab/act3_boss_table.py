"""Which Act 3 boss a run met, and what the health-bar reading did there (the Act 3 week's lane 1).

The judge's old kill reading took the first Act 3 boss as beaten when the fight's damage fraction reached 1.0. This tool tabulates, per first
Act 3 boss, the judged arms' runs that began its fight: how many went past it (`keep_v2.went_past`), how many read 1.0, and how many read 1.0
and still died there. The boss of a seed is read from a fresh emulator run session and the debug act jump (`Sts2Run_DebugEnterNextAct`,
list 14 of the run state): the acts are generated up front, so the first and second Act 3 bosses depend on the seed alone (59 of 60 plain
runs' own saved maps agree; the 60th had a second Hive map of 18 rooms). No run is played: about 30 ms a seed, one process.

    D:/spireline/src/sts2/.venv/Scripts/python.exe -I tools/lab/act3_boss_table.py --agent D:/spireline/referee/agent --gens 16 17 18 \
        [--sample 20] [--out table.json]

Needs the native emulator library (`out/` of the fork tree, or STS2_LIB_PATH). Read-only on the agent folder.
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import sys
from collections.abc import Callable, Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.lab import keep_v2 as k2  # noqa: E402

Identity = Callable[[str], tuple[str, str | None]]  # seed -> (first Act 3 boss, second Act 3 boss)


def first_entry(row: Mapping[str, Any], act: int) -> float | None:
    """The first `boss_damage_acts` entry of `act` (None when the fight never began, or its damage is unknown)."""
    for e in row.get("boss_damage_acts") or []:
        if int(e[0]) == act:
            return None if e[1] is None else float(e[1])
    return None


def began(row: Mapping[str, Any], act: int = 2) -> bool:
    """Whether the row records a fight of the first boss of `act` (a win does too)."""
    return bool(row.get("win")) or any(int(e[0]) == act for e in row.get("boss_damage_acts") or [])


def tabulate(rows: Iterable[Mapping[str, Any]], identity: Identity) -> dict[str, dict[str, int]]:
    """Per first Act 3 boss: arrivals (a fight of it began), past (the run went past it), bar_full (the first entry read 1.0), bar_full_died_there
    (read 1.0 and the run did not go past it), bar_part (the first entry read below 1.0), and per second boss of those past: arrived, wins."""
    out: dict[str, collections.Counter] = collections.defaultdict(collections.Counter)
    for r in rows:
        if r.get("crash") or r.get("lost") or not began(r):
            continue
        first, second = identity(str(r["seed"]))
        t = out[first]
        past = k2.went_past(r, 2)
        t["arrivals"] += 1
        t["past"] += past
        f = first_entry(r, 2)
        if f is not None and f >= 1.0:
            t["bar_full"] += 1
            t["bar_full_died_there"] += not past
        elif f is not None:
            t["bar_part"] += 1
        if past and second:
            t[f"second_{second}_arrived"] += 1
            t[f"second_{second}_wins"] += bool(r.get("win"))
    return {k: dict(v) for k, v in out.items()}


def engine_identity(lib: str | None = None) -> Identity:
    """`Identity` from the emulator: a fresh Ascension 10 Ironclad run session and the debug act jump to Act 3 (the native library is needed)."""
    if lib:
        os.environ["STS2_LIB_PATH"] = lib
    elif "STS2_LIB_PATH" not in os.environ and (ROOT / "out").is_dir():
        os.environ["STS2_LIB_PATH"] = str(ROOT / "out")
    os.environ.setdefault("STS2_ALLOW_STALE_NATIVE", "1")
    for p in (ROOT / "rl", ROOT / "src", ROOT):
        if str(p) not in sys.path:
            sys.path.insert(0, str(p))
    from run_train import engine, vocab  # noqa: PLC0415
    from sts2_gym import native  # noqa: PLC0415

    cache: dict[str, tuple[str, str | None]] = {}

    def identity(seed: str) -> tuple[str, str | None]:
        if seed not in cache:
            s = engine.RunSession(seed, 10)
            try:
                for _ in range(2):
                    s.env.debug_enter_next_act()
                lst = native.run_state_list(s.handle, 14, 4)
                cache[seed] = (vocab.BOSS_SLUGS.get(lst[1], str(lst[1])), vocab.BOSS_SLUGS.get(lst[3], str(lst[3])) if lst[3] else None)
            finally:
                native.run_destroy(s.handle)
        return cache[seed]

    return identity


def arm_rows(agent: Path, gens: Sequence[int]) -> list[tuple[str, dict[str, Any]]]:
    """(arm's relative path, row) of every judged arm run of the given generations' candidate folders."""
    out = []
    for g in gens:
        for p in sorted((Path(agent)).glob(f"gen{g}/*/arms/*-a3-v1/runs.jsonl")):
            rel = p.relative_to(agent).parent.as_posix()
            for line in p.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    out.append((rel, json.loads(line)))
    return out


def sample_died_full(rows: Sequence[tuple[str, Mapping[str, Any]]], identity: Identity, n: int, gens: Sequence[int]) -> list[dict[str, Any]]:
    """The first `n` rows (by arm path, then seed) of the generations in `gens` that read 1.0 at the first Act 3 boss, ended on floor 48 and died."""
    pick = []
    for rel, r in sorted(rows, key=lambda x: (x[0], str(x[1].get("seed")))):
        f = first_entry(r, 2)
        if f is not None and f >= 1.0 and not k2.went_past(r, 2) and r.get("end") == "death" and r.get("floor") == 48 \
                and any(rel.startswith(f"gen{g}/") for g in gens):
            first, second = identity(str(r["seed"]))
            pick.append({"arm": rel, "floor": r["floor"], "boss": first, "second_boss": second, "fraction": f})
        if len(pick) >= n:
            break
    return pick


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--agent", required=True)
    ap.add_argument("--gens", type=int, nargs="+", default=[16, 17, 18])
    ap.add_argument("--sample", type=int, default=0, help="also list this many runs that read 1.0 and died on floor 48 (no seeds are printed)")
    ap.add_argument("--lib", help="the native library folder (STS2_LIB_PATH)")
    ap.add_argument("--out")
    a = ap.parse_args(argv)
    ident = engine_identity(a.lib)
    rows = arm_rows(Path(a.agent), a.gens)
    doc: dict[str, Any] = {"gens": a.gens, "runs": len(rows), "by_first_boss": tabulate((r for _, r in rows), ident)}
    for boss, t in sorted(doc["by_first_boss"].items()):
        print(boss, t)
    if a.sample:
        s = sample_died_full(rows, ident, a.sample, a.gens)
        doc["sample"] = s
        print(collections.Counter(x["boss"] for x in s))
    if a.out:
        Path(a.out).write_text(json.dumps(doc, indent=1), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
