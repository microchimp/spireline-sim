"""Backfill the real-game record fields of a stored block from the game logs of its workers (A11).

Generation 3 and 5's real-game blocks (`referee/agent/gen*/real-s3-core2`) and the nightly block were recorded before the lockstep
recognised the boss fight (the boss node is the only choice left, so the real stream asks no map decision there). The rows cannot be
re-recorded without replaying, but each worker's `scratch/*/game-stdout.log` holds the driver's decision lines, enough for:

  boss fights   "Creating NCombatRoom ... encounter=X_BOSS" names every boss fight of the run, in order (the act from the boss), and
                the run's final floor says whether the last one was won (a floor past that act's boss floor). A boss never reached is
                0.0, a boss won is 1.0; a boss lost has no logged HP, so its damage is unknown (null), not guessed.
  rests         `run decision N rest "HEAL"`: every rest choice of the run, in order, "proceed" being the screen's leave. The stored
                row's rests carry the floors (the log has none), so a row whose count agrees with the log gets the log's kinds in
                its place; a row whose count differs keeps its own and is counted.

The log carries no boss HP and no floors, so nothing else is recomputed. Run: python rl/run_fields_backfill.py BLOCK_DIR [...]
Each block's runs.jsonl is copied to runs.jsonl.pre-a11.bak first (never overwritten), and the run_fields tables of summary.json
and compare.json are rebuilt from the new rows (rl.run_fields.aggregate).
"""

from __future__ import annotations

import json
import re
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "rl"))
sys.path.insert(0, str(ROOT))
import run_fields  # noqa: E402

from tools.lab import keep_v2  # noqa: E402  (the one strict rule for "the run went past the boss", `keep_v2.went_past`)

# The act (0-based) of each boss encounter (rl/run_train/vocab.BOSSES by region: Overgrowth and Underdocks act 1, Hive act 2, Glory act 3).
BOSS_ACT = {"CEREMONIAL_BEAST_BOSS": 0, "THE_KIN_BOSS": 0, "VANTOM_BOSS": 0, "LAGAVULIN_MATRIARCH_BOSS": 0, "SOUL_FYSH_BOSS": 0,
            "WATERFALL_GIANT_BOSS": 0, "KAISER_CRAB_BOSS": 1, "KNOWLEDGE_DEMON_BOSS": 1, "THE_INSATIABLE_BOSS": 1,
            "AEONGLASS_BOSS": 2, "QUEEN_BOSS": 2, "TEST_SUBJECT_BOSS": 2}
START = re.compile(r"^\[referee\] run decision 0 start")
REST = re.compile(r'^\[referee\] run decision \d+ rest "([^"]*)"')
BOSS = re.compile(r"^\[INFO\] Creating NCombatRoom .*encounter=([A-Z0-9_]+_BOSS)\.")
ENDED = re.compile(r"^\[referee\] run ([A-Z0-9]+) ended: ")


def parse_log(lines) -> dict[str, dict]:
    """seed -> {"rests": [kind, ...], "bosses": [encounter, ...]} for every run of one game log that ended."""
    out: dict[str, dict] = {}
    cur = {"rests": [], "bosses": []}
    for ln in lines:
        if START.match(ln):
            cur = {"rests": [], "bosses": []}
        elif m := REST.match(ln):
            if m.group(1).lower() not in run_fields.LEAVE_LABELS:
                cur["rests"].append(m.group(1).lower())
        elif m := BOSS.match(ln):
            cur["bosses"].append(m.group(1))
        elif m := ENDED.match(ln):
            out[m.group(1)] = cur
            cur = {"rests": [], "bosses": []}
    return out


def boss_fights(row: dict, bosses: list[str]) -> list[list]:
    """[[act, fraction | None], ...] of a run's boss fights: 1.0 for one won, None for the one lost (HP not logged). A fight is won when
    a later fight was logged, the run won, or it is the first boss of its act and the run went past it (`keep_v2.went_past`, the one
    strict rule: the floor above the boss's floor); the second Act 3 boss, lost or won, is a win of the run or a loss."""
    fights = []
    seen = [0] * len(run_fields.ACT_BOSS_FLOORS)
    for i, name in enumerate(bosses):
        act = BOSS_ACT.get(name)
        if act is None:
            continue
        last = i == len(bosses) - 1
        first_of_act = seen[act] == 0
        seen[act] += 1
        won = (not last) or row.get("end") == "win" or (first_of_act and keep_v2.went_past(row, act))
        fights.append([act, 1.0 if won else None])
    return fights


def backfill_row(row: dict, info: dict) -> tuple[dict, dict]:
    """The row with boss_damage, boss_damage_acts and rests rebuilt from its log info, and what was done."""
    new = dict(row)
    acts = boss_fights(row, info["bosses"])
    first = next((f for a, f in acts if a == 0), 0.0) if acts else 0.0
    new["boss_damage"], new["boss_damage_acts"] = first, acts
    kinds, rests = info["rests"], row.get("rests") or []
    note = {"rests": "kept"}
    if len(kinds) == len(rests):
        new["rests"] = [[floor, kind] for (floor, _old), kind in zip(rests, kinds)]
        note["rests"] = "changed" if new["rests"] != rests else "same"
    return new, note


def find_logs(block: Path) -> list[Path]:
    return sorted(block.glob("scratch/*/game-stdout.log"))


def backfill_block(block: Path) -> dict:
    """Rewrite one block's runs.jsonl, summary.json and compare.json; returns the counts the report needs."""
    runs = block / "runs.jsonl"
    infos: dict[str, dict] = {}
    for log in find_logs(block):
        with open(log, encoding="utf-8", errors="replace") as f:
            infos.update(parse_log(line.rstrip("\n") for line in f))
    rows = [json.loads(line) for line in runs.read_text(encoding="utf-8").splitlines() if line.strip()]
    stats = {"rows": 0, "logged": 0, "missing_log": 0, "rests_same": 0, "rests_changed": 0, "rests_kept": 0}
    out = []
    for r in rows:
        if r.get("ev") == "end" and "boss_damage" in r:
            stats["rows"] += 1
            info = infos.get(r.get("seed"))
            if info is None:
                stats["missing_log"] += 1
            else:
                stats["logged"] += 1
                r, note = backfill_row(r, info)
                stats["rests_" + note["rests"]] += 1
        out.append(r)
    ends = [r for r in out if r.get("ev") == "end" and "boss_damage" in r]
    bak = runs.with_name(runs.name + ".pre-a11.bak")
    if stats["logged"] and not bak.exists():
        shutil.copyfile(runs, bak)
    if stats["logged"]:
        runs.write_text("".join(json.dumps(r) + "\n" for r in out), encoding="utf-8", newline="\n")
        agg = run_fields.aggregate(ends)
        for name, path in (("summary.json", ("run_fields",)), ("compare.json", ("real", "run_fields"))):
            p = block / name
            doc = json.loads(p.read_text(encoding="utf-8"))
            tgt = doc
            for k in path[:-1]:
                tgt = tgt[k]
            if path[-1] in tgt:
                if not (bak.with_name(name + ".pre-a11.bak")).exists():
                    shutil.copyfile(p, bak.with_name(name + ".pre-a11.bak"))
                tgt[path[-1]].update({k: v for k, v in agg.items() if k in ("boss_damage", "boss_act1")})
                p.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8", newline="\n")
    stats["boss_act1"] = run_fields.aggregate(ends).get("boss_act1")
    stats["boss_damage"] = run_fields.aggregate(ends).get("boss_damage")
    kinds: dict[str, int] = {}
    for r in ends:
        for _f, k in r.get("rests") or []:
            kinds[k] = kinds.get(k, 0) + 1
    stats["rest_kinds"] = kinds
    return stats


def main(argv: list[str]) -> int:
    for b in argv:
        print(b, json.dumps(backfill_block(Path(b))))
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
