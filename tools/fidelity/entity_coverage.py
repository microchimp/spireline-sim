"""Per-entity lab coverage for the logic audit: which cards, relics, powers and enemies the lab's fights exercised,
and how many of those fights matched the real game.

    python -m tools.fidelity.entity_coverage [--runs DIR] [--decompiled DIR] [--out DIR] [--min-fights 30]

Input: the lab's fight records, <runs>/*/fights/*.out.json (our replay of each fight, state per step) and the
sibling .verdicts.json (identical or first divergence, with the diverging card and field diffs). The records hold
no per-effect trace event, so "triggered" is PRESENCE, by kind:

  card   played: a 'play' step whose card is the id, with played true;
  relic  in the player's relic list of any step (the state does not say when a relic fired);
  power  in the player's or an enemy's power list of any step;
  enemy  in the enemies list of any step (so it acted whenever the fight went past the start).

Per entity: fights (triggered), identical (fight matched the game end to end), diverged, attributed (diverged fights
where the diverging card is the entity, or its id occurs in the non-pile field diffs: a card id in a hand or pile list does not count), clean_before (diverged fights
where it was present before the first divergence and not attributed: those steps matched). Well covered =
fights >= --min-fights and attributed == 0. The universe is the Ironclad and colourless card pools plus every
relic, power and monster class of the local decompile, plus anything seen in a fight.

Writes <out>/entity-coverage.csv and prints a summary. Reads game data from outside the repo; writes nothing in it.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
import sys
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

DEFAULT_RUNS = Path("D:/spireline/referee/lab/runs")
DEFAULT_DECOMPILED = Path("D:/spireline/referee/decompiled")
DEFAULT_OUT = Path("D:/spireline/referee/audit/data-checks")
MODELS = "MegaCrit.Sts2.Core.Models"


def snake(name: str) -> str:
    return re.sub(r"(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_", name).upper()


def _ids(x) -> list[str]:
    out = []
    for e in x or []:
        if isinstance(e, str):
            out.append(e)
        elif isinstance(e, (list, tuple)) and e:
            out.append(str(e[0]))
        elif isinstance(e, dict):
            out.append(str(e.get("id") or e.get("name") or ""))
    return [i for i in out if i]


def scan_file(path: str) -> list[tuple]:
    """One tuple per fight: (kind, first_div_step|None, identical, attribution text, diverging card, {entity: first step})."""
    p = Path(path)
    try:
        res = json.loads(p.read_text(encoding="utf-8"))["results"]
        ver = json.loads(p.with_name(p.name.replace(".out.json", ".verdicts.json")).read_text(encoding="utf-8"))
    except (OSError, ValueError, KeyError):
        return []
    vmap = {v.get("seed"): v for v in ver if isinstance(v, dict)}
    rows = []
    for i, f in enumerate(res):
        v = vmap.get(f.get("seed")) or (ver[i] if len(ver) == len(res) and isinstance(ver[i], dict) else {})
        first: dict[str, int] = {}
        for si, s in enumerate(f.get("steps", [])):
            a = s.get("action") or {}
            st = s.get("state") or {}
            ents = []
            if a.get("kind") == "play" and a.get("played", True) and a.get("card"):
                ents.append("card:" + a["card"])
            ents += ["relic:" + r for r in _ids(st.get("relics"))]
            ents += ["power:" + r for r in _ids((st.get("player") or {}).get("powers"))]
            for e in st.get("enemies") or []:
                ents.append("enemy:" + str(e.get("name")))
                ents += ["power:" + r for r in _ids(e.get("powers"))]
            for e in ents:
                first.setdefault(e, si)
        identical = bool(v.get("identical")) if v else True
        piles = (".hand", ".draw", ".discard", ".exhaust")
        text = json.dumps([d for d in v.get("diff") or [] if not str(d[0]).startswith(piles)]) if not identical else ""
        rows.append(((f.get("fight") or {}).get("kind", ""), None if identical else v.get("step"), identical,
                     text, v.get("card"), first))
    return rows


def universe(dec: Path) -> dict[str, str]:
    u: dict[str, str] = {}
    for pool in ("Ironclad", "Colorless"):
        f = dec / (MODELS + ".CardPools") / f"{pool}CardPool.cs"
        if f.exists():
            for c in re.findall(r"Card<(\w+)>", f.read_text(encoding="utf-8", errors="replace")):
                u["card:" + snake(c)] = "card"
    for sub, kind in (("Relics", "relic"), ("Powers", "power"), ("Monsters", "enemy")):
        d = dec / f"{MODELS}.{sub}"
        if d.is_dir():
            for f in d.glob("*.cs"):
                u[f"{kind}:{snake(f.stem)}"] = kind
    return u


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--runs", type=Path, default=DEFAULT_RUNS)
    ap.add_argument("--decompiled", type=Path, default=DEFAULT_DECOMPILED)
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT)
    ap.add_argument("--min-fights", type=int, default=30)
    ap.add_argument("--jobs", type=int, default=6)
    a = ap.parse_args()
    files = sorted(str(p) for p in a.runs.glob("*/fights/*.out.json") if ".shrink" not in p.name)
    blank = dict(fights=0, identical=0, diverged=0, attributed=0, clean_before=0, boss=0)
    stat: dict[str, dict] = defaultdict(lambda: dict(blank))
    nfights = 0
    with ProcessPoolExecutor(a.jobs) as ex:
        for rows in ex.map(scan_file, files, chunksize=4):
            for kind, step, identical, text, card, first in rows:
                nfights += 1
                for e, si in first.items():
                    s = stat[e]
                    s["fights"] += 1
                    if "BOSS" in kind.upper() or "ELITE" in kind.upper():
                        s["boss"] += 1
                    if identical:
                        s["identical"] += 1
                        continue
                    s["diverged"] += 1
                    bare = e.split(":", 1)[1]
                    if card == bare or bare in text:
                        s["attributed"] += 1
                    elif step is None or si < step:
                        s["clean_before"] += 1
    uni = universe(a.decompiled)
    for e in stat:
        uni.setdefault(e, e.split(":")[0])
    a.out.mkdir(parents=True, exist_ok=True)
    path = a.out / "entity-coverage.csv"
    cols = ["entity", "kind", "fights", "identical", "diverged", "attributed", "clean_before", "elite_or_boss_fights", "status"]
    tally: dict[str, list[int]] = defaultdict(lambda: [0, 0, 0])
    with path.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(cols)
        for e in sorted(uni, key=lambda k: (uni[k], k)):
            s = stat.get(e, blank)
            ok = s["fights"] >= a.min_fights and s["attributed"] == 0
            status = "well-covered" if ok else ("attributed-divergence" if s["attributed"] else ("thin" if s["fights"] else "unseen"))
            tally[uni[e]][0 if ok else (1 if s["fights"] else 2)] += 1
            w.writerow([e, uni[e], s["fights"], s["identical"], s["diverged"], s["attributed"], s["clean_before"], s["boss"], status])
    print(f"{nfights} fights from {len(files)} record files; 'triggered' = presence in the step states (no trace event field)")
    for k, (g, t, z) in sorted(tally.items()):
        print(f"{k}: {g + t + z} entities, {g} well covered (>={a.min_fights} fights, 0 attributed divergences), "
              f"{t} seen but need reading, {z} never seen")
    print(f"wrote {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
