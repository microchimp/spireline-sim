"""Re-score every saved judgement under the strict boss reading (2026-11-19, the Act 3 week's lane 1).

The judge used to read a boss as beaten when `boss_damage_acts` had a first entry of 1.0 for its act (`keep_v2.bosses_beaten_loose`). At the
first Act 3 boss that is about three times too high: the Test Subject's first form is emptied in 97 of 100 runs that meet it, and the boss
then comes back twice (docs/rl/KEEP_TEST_V2.md). The strict reading (`keep_v2.bosses_beaten`) counts a boss beaten when the run went past it.

This tool reads the ledger (`<agent>/promotions.tsv`) and, for each judgement whose two arms are on disk, counts per arm B1, B2, B3a and wins
under both readings, and writes one row per judgement (`--out`, a tab-separated file; the ledger itself is never written). With
`--generations` it also writes one row per promoted generation: all the runs its agent played in judged arms (as candidate and as
incumbent), loose and strict. Read-only on the agent folder; one process, no emulator; a few seconds.

    python -I tools/lab/ledger_strict.py --agent D:/spireline/referee/agent --out ledger-strict.tsv --generations generations-strict.tsv

Arms are found as the judge found them: the candidate's `gen<cand_gen>/<cand_dir>/arms/<block>-a3-v1` and the incumbent's under the folder of
the base generation's promoted candidate (`gen0/arms/<block>-a3-v1` for generation 0). A pooled judgement (the ledger's `guard` cell says
"pooled <block> (n) + <block> (n)") pools both blocks. Only the seeds both arms played count (`pairs`). Rows from before 2026-10-25 carry no
`boss_damage_acts`: both readings are then the floor cuts and agree.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections.abc import Iterable, Mapping, Sequence
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.lab import keep_v2 as k2  # noqa: E402

CONFIG = "a3-v1"
POOLED = re.compile(r"pooled (\S+) \((\d+)\) \+ (\S+) \((\d+)\)")
BOSS_NAMES = ("B1", "B2", "B3a", "win")
LEDGER_COLUMNS = ("written", "cand_gen", "cand_dir", "base_gen", "block", "pairs",
                  "b1_loose_cand", "b1_loose_base", "b1_strict_cand", "b1_strict_base",
                  "b2_loose_cand", "b2_loose_base", "b2_strict_cand", "b2_strict_base",
                  "b3a_loose_cand", "b3a_loose_base", "b3a_strict_cand", "b3a_strict_base",
                  "wins_cand", "wins_base", "b3a_strict_diff_per100", "b3a_strict_se_per100", "decision", "rule", "route", "arms")
GEN_COLUMNS = ("gen", "agent_dir", "arms", "runs", "b1_loose", "b1_strict", "b2_loose", "b2_strict", "b3a_loose", "b3a_strict", "wins",
               "b3a_loose_per100", "b3a_strict_per100", "wins_per100", "crashes")


def read_ledger(agent: Path) -> list[dict[str, str]]:
    """The rows of `promotions.tsv` by column name (a short row has empty cells)."""
    tsv = Path(agent) / "promotions.tsv"
    lines = [x for x in tsv.read_text(encoding="utf-8").splitlines() if x.strip()]
    head = lines[0].split("\t")
    return [dict(zip(head, [*x.split("\t"), *[""] * len(head)], strict=False)) for x in lines[1:]]


def read_arm(path: Path) -> dict[str, dict[str, Any]]:
    """An arm's rows by seed."""
    out: dict[str, dict[str, Any]] = {}
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            r = json.loads(line)
            out[r["seed"]] = r
    return out


def promoted_dir(rows: Sequence[Mapping[str, str]], gen: str) -> str:
    """The folder name (under `gen<N>`) of the candidate the ledger promoted into generation `gen` (the last such row)."""
    found = [r for r in rows if r["cand_gen"] == gen and r["decision"] == "promoted"]
    return (found[-1]["cand_dir"] or "candidate") if found else "candidate"


def arm_dirs(agent: Path, rows: Sequence[Mapping[str, str]], row: Mapping[str, str], block: str) -> tuple[Path, Path]:
    """(candidate's arm folder, incumbent's arm folder) of one block of one judgement."""
    cand = Path(agent) / f"gen{row['cand_gen']}" / (row["cand_dir"] or "candidate") / "arms" / f"{block}-{CONFIG}"
    if row["base_gen"] == "0":
        base = Path(agent) / "gen0" / "arms" / f"{block}-{CONFIG}"
    else:
        base = Path(agent) / f"gen{row['base_gen']}" / promoted_dir(rows, row["base_gen"]) / "arms" / f"{block}-{CONFIG}"
    return cand, base


def blocks_of(row: Mapping[str, str]) -> list[str]:
    """The blocks a judgement's pairs come from: the ledger's block, and for a pooled judgement both blocks its `guard` cell names."""
    m = POOLED.search(row.get("guard", ""))
    return [m.group(1), m.group(3)] if m else [row["seed_block"]]


def tally(rows: Iterable[Mapping[str, Any]], reading: Any) -> list[int]:
    """[B1, B2, B3a, wins] of the rows under `reading` (a `bosses_beaten`-like function)."""
    out = [0, 0, 0, 0]
    for r in rows:
        for j, beat in enumerate(reading(r)):
            out[j] += bool(beat)
    return out


def paired(cand: Mapping[str, Mapping[str, Any]], base: Mapping[str, Mapping[str, Any]]) -> tuple[list[Mapping[str, Any]], list[Mapping[str, Any]]]:
    seeds = sorted(set(cand) & set(base))
    return [cand[s] for s in seeds], [base[s] for s in seeds]


def diff_per100(c_rows: Sequence[Mapping[str, Any]], b_rows: Sequence[Mapping[str, Any]], j: int, reading: Any = k2.bosses_beaten) -> tuple[float, float]:
    """The paired difference in boss `j`'s kills per 100 runs and its standard error (the paired differences' own spread)."""
    n = len(c_rows)
    if n < 2:
        return 0.0, 0.0
    d = [float(reading(c)[j]) - float(reading(b)[j]) for c, b in zip(c_rows, b_rows, strict=True)]
    m = sum(d) / n
    var = sum((x - m) ** 2 for x in d) / (n - 1)
    return 100.0 * m, 100.0 * (var / n) ** 0.5


def rescore(agent: Path) -> list[dict[str, Any]]:
    """One row per ledger judgement whose arms are all on disk (`arms` says what was missing otherwise)."""
    ledger = read_ledger(agent)
    out = []
    for r in ledger:
        c_all: dict[str, Mapping[str, Any]] = {}
        b_all: dict[str, Mapping[str, Any]] = {}
        missing = []
        for block in blocks_of(r):
            cd, bd = arm_dirs(agent, ledger, r, block)
            if not (cd / "runs.jsonl").is_file() or not (bd / "runs.jsonl").is_file():
                missing.append(block)
                continue
            c, b = read_arm(cd / "runs.jsonl"), read_arm(bd / "runs.jsonl")
            for s in set(c) & set(b):  # a seed is judged once: pooled blocks hold no seed twice
                c_all[f"{block}:{s}"], b_all[f"{block}:{s}"] = c[s], b[s]
        row: dict[str, Any] = {"written": r["written"], "cand_gen": r["cand_gen"], "cand_dir": r["cand_dir"] or "candidate",
                               "base_gen": r["base_gen"], "block": r["seed_block"], "decision": r["decision"], "rule": r.get("rule") or "v1",
                               "route": r.get("route", ""), "arms": "missing: " + ",".join(missing) if missing else "ok"}
        if missing or not c_all:
            out.append(row)
            continue
        c_rows, b_rows = paired(c_all, b_all)
        lo_c, lo_b = tally(c_rows, k2.bosses_beaten_loose), tally(b_rows, k2.bosses_beaten_loose)
        st_c, st_b = tally(c_rows, k2.bosses_beaten), tally(b_rows, k2.bosses_beaten)
        d, se = diff_per100(c_rows, b_rows, 2)
        row.update(pairs=len(c_rows), b3a_strict_diff_per100=round(d, 2), b3a_strict_se_per100=round(se, 2), wins_cand=st_c[3], wins_base=st_b[3])
        for j, name in enumerate(("b1", "b2", "b3a")):
            row[f"{name}_loose_cand"], row[f"{name}_loose_base"] = lo_c[j], lo_b[j]
            row[f"{name}_strict_cand"], row[f"{name}_strict_base"] = st_c[j], st_b[j]
        out.append(row)
    return out


def generations(agent: Path) -> list[dict[str, Any]]:
    """One row per promoted generation: every run its agent played in a judged arm (as candidate and as incumbent), loose and strict."""
    ledger = read_ledger(agent)
    out = []
    seen = set()
    for r in ledger:
        if r["decision"] != "promoted" or r["cand_gen"] in seen:
            continue
        seen.add(r["cand_gen"])
    for g in sorted(seen, key=int):
        d = promoted_dir(ledger, g)
        arms = sorted((Path(agent) / f"gen{g}" / d / "arms").glob(f"*-{CONFIG}/runs.jsonl"))
        rows = [r for p in arms for r in read_arm(p).values()]
        if not rows:
            continue
        lo, st = tally(rows, k2.bosses_beaten_loose), tally(rows, k2.bosses_beaten)
        n = len(rows)
        out.append({"gen": int(g), "agent_dir": d, "arms": len(arms), "runs": n, "b1_loose": lo[0], "b1_strict": st[0], "b2_loose": lo[1],
                    "b2_strict": st[1], "b3a_loose": lo[2], "b3a_strict": st[2], "wins": st[3], "b3a_loose_per100": round(100 * lo[2] / n, 2),
                    "b3a_strict_per100": round(100 * st[2] / n, 2), "wins_per100": round(100 * st[3] / n, 3),
                    "crashes": sum(bool(r.get("crash")) for r in rows)})
    return out


def write_tsv(path: Path, columns: Sequence[str], rows: Iterable[Mapping[str, Any]]) -> None:
    Path(path).write_text("\t".join(columns) + "\n" + "".join("\t".join(str(r.get(c, "")) for c in columns) + "\n" for r in rows), encoding="utf-8")


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--agent", required=True, help="the agent folder (promotions.tsv and gen*/)")
    ap.add_argument("--out", required=True, help="the per-judgement table (tab-separated)")
    ap.add_argument("--generations", help="the per-generation table (tab-separated)")
    a = ap.parse_args(argv)
    rows = rescore(Path(a.agent))
    write_tsv(Path(a.out), LEDGER_COLUMNS, rows)
    print(f"{len(rows)} judgements, {sum(1 for r in rows if r['arms'] == 'ok')} with their arms on disk -> {a.out}")
    if a.generations:
        g = generations(Path(a.agent))
        write_tsv(Path(a.generations), GEN_COLUMNS, g)
        print(f"{len(g)} generations -> {a.generations}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
