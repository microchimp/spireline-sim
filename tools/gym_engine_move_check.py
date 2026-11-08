#!/usr/bin/env python3
"""Did the engine move change the fight player's games? The check D51 made by hand when the fight loop moved to a new engine tag.

    python tools/gym_engine_move_check.py GAMES_A GAMES_B [--list N] [--json] [--config NAME]

GAMES_A and GAMES_B are two games files of the same player on the same starts under two engines, in the gym's row format (the gate's
`gate/inc-<player>-<config>-<engine>.jsonl`, the exam's `exam/games-<player>.jsonl`, any `fight_gym` rows): one JSON object a line,
with `start`, `j`, `won`, `capped`, `end_hp` and `moves_sha` (the hash of the moves played). Games are matched by (start, j); a game
played twice keeps its last row. It prints, for the matched games:

  * the counts: rows in each file, matched, only in A, only in B;
  * outcome agreement (`won` and not `capped` the same), end-HP agreement (`end_hp`), and move-hash agreement (`moves_sha`, over the
    games where both rows have one), each as a count and a share; and the games whose moves differ though outcome and end HP agree;
  * the root value's largest and median difference (`root_value`, when both rows have it): float noise in the search shows as ~1e-7;
  * the engine tags the rows (or, for a gate file, its name) carry.

`--list N` names up to N games that differ (start, j and what differs); `--json` prints the figures as one JSON object. Standard
library only; it reads two files and writes nothing. Exit code 0, or 2 when no game is matched.
"""

from __future__ import annotations

import argparse
import json
import re
import statistics
import sys
from pathlib import Path


def read_rows(path: Path) -> list[dict]:
    """The JSON object rows of a games file; a line a stop cut short is skipped."""
    rows = []
    with open(path, encoding="utf-8") as fh:
        for line in fh:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
            except ValueError:
                continue
            if isinstance(row, dict) and "start" in row and "j" in row:
                rows.append(row)
    return rows


def index(rows: list[dict], config: str | None = None) -> dict[tuple[str, int], dict]:
    """{(start, j): row}, the last row of a game kept; with `config` only rows played under that config."""
    return {(r["start"], int(r["j"])): r for r in rows if config is None or r.get("config", config) == config}


def won(r: dict) -> bool:
    return bool(r.get("won")) and not r.get("capped")


def engines(rows: list[dict], path: Path) -> list[str]:
    """The engine tags in the rows, else the tag a gate file's name ends with (`...-<12 hex>.jsonl`)."""
    tags = sorted({str(r["engine"]) for r in rows if r.get("engine")})
    if tags:
        return tags
    m = re.search(r"-([0-9a-f]{12})\.jsonl$", Path(path).name)
    return [m.group(1)] if m else []


def compare(a: dict, b: dict) -> dict:
    """The agreement of two indexed games files on the games they share."""
    keys = sorted(set(a) & set(b))
    out: dict = {"rows_a": len(a), "rows_b": len(b), "matched": len(keys), "only_a": len(set(a) - set(b)), "only_b": len(set(b) - set(a))}
    diffs: list[dict] = []
    outcome = hp = moves = moves_n = moves_only = 0
    root: list[float] = []
    for k in keys:
        x, y = a[k], b[k]
        d: dict = {}
        if won(x) == won(y):
            outcome += 1
        else:
            d["outcome"] = [won(x), won(y)]
        if x.get("end_hp") == y.get("end_hp"):
            hp += 1
        else:
            d["end_hp"] = [x.get("end_hp"), y.get("end_hp")]
        if x.get("moves_sha") is not None and y.get("moves_sha") is not None:
            moves_n += 1
            if x["moves_sha"] == y["moves_sha"]:
                moves += 1
            else:
                d["moves_sha"] = [x["moves_sha"], y["moves_sha"]]
                if "outcome" not in d and "end_hp" not in d:
                    moves_only += 1
        if isinstance(x.get("root_value"), (int, float)) and isinstance(y.get("root_value"), (int, float)):
            root.append(abs(float(x["root_value"]) - float(y["root_value"])))
        if d:
            diffs.append({"start": k[0], "j": k[1], **d})
    n = len(keys)

    def share(c: int, of: int) -> float | None:
        return None if of == 0 else c / of

    out.update(outcome_agree=outcome, outcome_share=share(outcome, n), end_hp_agree=hp, end_hp_share=share(hp, n),
               moves_compared=moves_n, moves_agree=moves, moves_share=share(moves, moves_n), moves_differ_only=moves_only,
               win_a=sum(1 for k in keys if won(a[k])), win_b=sum(1 for k in keys if won(b[k])),
               root_value_pairs=len(root), root_value_max_diff=max(root) if root else None,
               root_value_median_diff=statistics.median(root) if root else None, differing=diffs)
    return out


def text(res: dict, name_a: str, name_b: str, tags_a: list[str], tags_b: list[str], listing: int) -> str:
    def f(c, n, s):
        return f"{c} of {n}" + ("" if s is None else f" ({100 * s:.1f}%)")

    m = res["matched"]
    lines = [f"A {name_a}: {res['rows_a']} games, engine {','.join(tags_a) or 'not named'}",
             f"B {name_b}: {res['rows_b']} games, engine {','.join(tags_b) or 'not named'}",
             f"matched starts (start, j): {m}; only in A {res['only_a']}, only in B {res['only_b']}",
             f"wins: A {res['win_a']}, B {res['win_b']} of {m}",
             f"outcome agreement:   {f(res['outcome_agree'], m, res['outcome_share'])}",
             f"end-HP agreement:    {f(res['end_hp_agree'], m, res['end_hp_share'])}",
             f"move-hash agreement: {f(res['moves_agree'], res['moves_compared'], res['moves_share'])}"
             + (f"; {res['moves_differ_only']} of the games whose moves differ have the same outcome and end HP" if res["moves_compared"] else "")]
    if res["root_value_pairs"]:
        lines.append(f"root value: largest difference {res['root_value_max_diff']:.3g}, median {res['root_value_median_diff']:.3g} "
                     f"over {res['root_value_pairs']} games")
    if listing and res["differing"]:
        lines.append(f"differing games (first {min(listing, len(res['differing']))} of {len(res['differing'])}):")
        for d in res["differing"][:listing]:
            lines.append("  " + d["start"] + f" j={d['j']}: " + ", ".join(f"{k} {v[0]} -> {v[1]}" for k, v in d.items() if k not in ("start", "j")))
    return "\n".join(lines)


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("games_a")
    ap.add_argument("games_b")
    ap.add_argument("--list", type=int, default=0, metavar="N", help="name up to N games that differ")
    ap.add_argument("--json", action="store_true", help="print the figures as one JSON object")
    ap.add_argument("--config", default=None, help="compare only rows played under this config")
    a = ap.parse_args(argv)
    pa, pb = Path(a.games_a), Path(a.games_b)
    rows_a, rows_b = read_rows(pa), read_rows(pb)
    res = compare(index(rows_a, a.config), index(rows_b, a.config))
    tags_a, tags_b = engines(rows_a, pa), engines(rows_b, pb)
    if a.json:
        print(json.dumps({**res, "engine_a": tags_a, "engine_b": tags_b, "differing": res["differing"][: a.list] if a.list else []}, indent=1))
    else:
        print(text(res, pa.name, pb.name, tags_a, tags_b, a.list))
    return 0 if res["matched"] else 2


if __name__ == "__main__":
    sys.exit(main())
