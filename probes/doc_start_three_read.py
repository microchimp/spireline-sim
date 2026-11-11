"""Read doc_start_three.py's files: survival and HP of the ways played (A the run clone on the run handle, B a run fork through the gym loop,
C the document), paired by door; the differences with 95% bootstrap intervals over doors (the door is the unit, as in doc_vs_run_read.py).

    python -I doc_start_three_read.py DIR [--json OUT]

HP is the fight's own end HP (0 for a death) for every way; A's HP before that correction (the run block's, with the rewards screen's heal) is
printed as `hp_run_block`.
"""
import json
import random
import sys
from pathlib import Path


def boot(xs, n=4000, seed=1):
    """95% bootstrap interval of the mean of door-level differences."""
    rnd = random.Random(seed)
    ms = sorted(sum(rnd.choice(xs) for _ in xs) / len(xs) for _ in range(n))
    return ms[int(0.025 * n)], ms[int(0.975 * n)]


def main():
    doors = {}
    for p in sorted(Path(sys.argv[1]).glob("three_*.json")):
        d = json.loads(p.read_text())
        doors.setdefault((d["seed"], d["floor"]), []).extend(d["games"])
    ways = [w for w in "ABC" if all(w in g for gs in doors.values() for g in gs)]
    print("doors", len(doors), "games per way", sum(len(gs) for gs in doors.values()), "ways", "".join(ways))
    per = {}
    for k, gs in sorted(doors.items()):
        row = {w: (sum(g[w]["won"] for g in gs) / len(gs) * 100, sum(g[w]["hp"] for g in gs) / len(gs), len(gs)) for w in ways}
        row["Ablock"] = sum(g["A"].get("hp_run_block", 0) for g in gs) / len(gs)
        per[k] = row
        print(k, " ".join(f"{w}: surv {row[w][0]:5.1f} hp {row[w][1]:5.1f}" for w in ways), "n", len(gs))
    out = {"doors": len(per), "ways": "".join(ways), "survival": {}, "hp": {}, "diff": {}}
    for w in ways:
        out["survival"][w] = round(sum(per[k][w][0] for k in per) / len(per), 1)
        out["hp"][w] = round(sum(per[k][w][1] for k in per) / len(per), 1)
    out["hp"]["A_run_block"] = round(sum(per[k]["Ablock"] for k in per) / len(per), 1)
    print("survival", out["survival"], "hp", out["hp"])
    for a, b in (("A", "C"), ("B", "C"), ("A", "B")):
        if a in ways and b in ways:
            s = [per[k][b][0] - per[k][a][0] for k in per]
            h = [per[k][b][1] - per[k][a][1] for k in per]
            out["diff"][f"{b}-{a}"] = {"survival": [round(sum(s) / len(s), 1), *[round(x, 1) for x in boot(s)]], "hp": [round(sum(h) / len(h), 1), *[round(x, 1) for x in boot(h)]]}
            print(f"{b} minus {a}: survival {out['diff'][f'{b}-{a}']['survival'][0]:+.1f} ({out['diff'][f'{b}-{a}']['survival'][1]:+.1f} to {out['diff'][f'{b}-{a}']['survival'][2]:+.1f})  "
                  f"hp {out['diff'][f'{b}-{a}']['hp'][0]:+.1f} ({out['diff'][f'{b}-{a}']['hp'][1]:+.1f} to {out['diff'][f'{b}-{a}']['hp'][2]:+.1f})")
    h = [per[k]["C"][1] - per[k]["Ablock"] for k in per] if "C" in ways else []
    if h:
        out["diff"]["C-A_run_block"] = {"hp": [round(sum(h) / len(h), 1), *[round(x, 1) for x in boot(h)]]}
        print("the probe's own reading (C minus A's run-block HP): hp", out["diff"]["C-A_run_block"]["hp"])
    if "--json" in sys.argv:
        Path(sys.argv[sys.argv.index("--json") + 1]).write_text(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
