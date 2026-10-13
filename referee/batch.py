#!/usr/bin/env python3
"""One fidelity batch end to end: make fights, play them in the game, replay them in the emulator, compare.

    python referee/batch.py --name <name> --fights 100 [--start 0] [--timeout 5400]

Run from the fork's root. Writes D:/spireline/referee/jobs/<name>.{json,out.json,emu.json,summary.txt}, <name>.packs/ (context_pack)
and prints the summary: fights identical, fights diverging grouped by the first differing field,
and errors. Seeds are deterministic from --start, so a batch can be re-run exactly.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
JOBS = Path(r"D:\spireline\referee\jobs")
GAME = r"D:\spireline\referee\game"
ALPHABET = "0123456789ABCDEFGHJKLMNPQRSTUVWXYZ"


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--name", required=True)
    ap.add_argument("--fights", type=int, default=100)
    ap.add_argument("--start", type=int, default=0, help="seed stream offset")
    ap.add_argument("--timeout", type=float, default=5400)
    ap.add_argument("--parallel", type=int, default=6, help="game copies running at once")
    ap.add_argument("--deck-cards", type=int, default=5, help="random Ironclad/colourless cards added per fight")
    ap.add_argument("--canary", type=int, default=2, help="fights per copy run and checked before the rest (0 = off)")
    ap.add_argument("--waves", type=int, default=3, help="waves per batch; coverage is updated between waves")
    a = ap.parse_args()

    # Coverage: each fight's deck gets random cards from the pools an Ironclad can hold, so a
    # batch exercises hundreds of cards rather than the starter deck's three.
    pools = json.loads(subprocess.run(["dotnet", str(ROOT / "tools/emudump/bin/Release/net9.0/EmuDump.dll"), "pools"],
                                      capture_output=True, text=True, check=True).stdout)
    id_map = json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8"))["cards"]
    sys.path.insert(0, str(ROOT))
    from tools.fidelity.fight_compare import snake  # noqa: PLC0415
    names = {v: snake(k) for k, v in id_map.items()}
    candidates = sorted({names[i] for i in pools["ironclad"] + pools["colorless"] if i in names})

    t_start = time.time()
    from tools.fidelity import coverage as cv  # noqa: PLC0415
    from tools.fidelity.fight_compare import names as id_names  # noqa: PLC0415
    from tools.fidelity import reanchor_compare as rc  # noqa: PLC0415
    nm = id_names(json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8")))
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    cov_path = JOBS.parent / "coverage.json"
    cov = cv.load(cov_path)
    rng = random.Random(a.start)

    def make_runs(n: int, offset: int) -> list:
        # Decks favour untested cards and leave out cards quarantined as suspect at this commit.
        return [{"kind": "combat", "seed": "".join(rng.choice(ALPHABET) for _ in range(10)), "character": "IRONCLAD",
                 "ascension": 10, "neow": (offset + i) % 3, "chooser_seed": a.start * 1000 + offset + i,
                 "max_actions": 150, "deck_add": cv.pick(cov, candidates, a.deck_cards, rng, commit)}
                for i in range(n)]

    runs = make_runs(a.fights, 0)  # replaced wave by wave below; kept for the job record
    job, out, emu = (JOBS / f"{a.name}{ext}" for ext in (".json", ".out.json", ".emu.json"))
    job.write_text(json.dumps({"out": str(out).replace("\\", "/"), "runs": runs}), encoding="utf-8")

    sys.path.insert(0, str(ROOT / "referee"))
    from run_referee import install  # noqa: PLC0415
    install(Path(GAME), ROOT / "referee" / "driver" / "bin" / "Release" / "net9.0")  # once, before any copy starts

    def play(name: str, subset: list) -> tuple[list, list]:
        """Run `subset` on up to --parallel long-lived game servers (serve.py); results in subset order.

        serve.py boots each game once, feeds it fights from a shared inbox, restarts a server that
        hangs or dies and re-queues its fight once; a fight that fails twice comes back as stalled.
        """
        spec, sout = JOBS / f"{name}.json", JOBS / f"{name}.out.json"
        run_dir = Path(GAME).parent / "serve" / name
        spec.write_text(json.dumps({"runs": subset}), encoding="utf-8")
        p = subprocess.run([sys.executable, str(ROOT / "referee" / "serve.py"), "--game", GAME, "--specs", str(spec),
                            "--servers", str(max(1, min(a.parallel, len(subset)))), "--out", str(sout),
                            "--run-dir", str(run_dir), "--no-install"],
                           stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=a.timeout, check=False)
        text = p.stdout
        try:  # serve.py's own summary.json (the stdout summary follows its log lines)
            rep = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
        except (FileNotFoundError, ValueError):
            rep = {"tail": text[-500:]}
        rep.update(exit=p.returncode, seconds=rep.get("wall_s"))
        got = json.loads(sout.read_text(encoding="utf-8"))["results"] if sout.exists() else []
        merged = [got[n] if n < len(got) and got[n] else {"seed": r["seed"], "error": "server produced no result", "steps": []}
                  for n, r in enumerate(subset)]
        return merged, [rep]

    def reanchor(out_path: Path, rea_path: Path) -> dict:
        """EmuDump reanchor: each recorded step replayed from the game's own state before it."""
        with open(rea_path, "w", encoding="utf-8") as f:
            subprocess.run(["dotnet", str(ROOT / "tools/emudump/bin/Release/net9.0/EmuDump.dll"), "reanchor", str(out_path)],
                           stdout=f, check=False)
        return json.loads(rea_path.read_text(encoding="utf-8"))

    def replay_and_learn(results: list, tag: str) -> int:
        """Replay a wave in the emulator, fold it into coverage, return identical count."""
        w_out, w_emu = JOBS / f"{tag}.out.json", JOBS / f"{tag}.emu.json"
        w_out.write_text(json.dumps({"results": results}), encoding="utf-8")
        with open(w_emu, "w", encoding="utf-8") as f:
            subprocess.run(["dotnet", str(ROOT / "tools/emudump/bin/Release/net9.0/EmuDump.dll"), "fight", str(w_out)],
                           stdout=f, check=False)
        emus = json.loads(w_emu.read_text(encoding="utf-8"))["results"]
        # Coverage learns from every re-anchored step (SPEED_PLAN step 6), not only those before
        # the first divergence; reanchor_compare.learn replaces coverage.update.
        rc.learn(cov, results, reanchor(w_out, JOBS / f"{tag}.reanchor.json"), nm, commit)
        cv.save(cov_path, cov)
        return sum(1 for g, e in zip(results, emus)
                   if g.get("steps") and not g.get("error") and cv.first_divergence(g, e, nm) is None)

    merged, reports = [], []
    remaining = a.fights
    # Canary: a few fights per copy first, compared before the rest is spent on a broken setup.
    if a.canary and remaining > a.canary * a.parallel:
        head = make_runs(a.canary * a.parallel, 0)
        c_res, c_rep = play(f"{a.name}.canary", head)
        bad = [r for r in c_res if r.get("error")]
        # Systemic only: a stalled fight is recorded and skipped, so a few are tolerated; any
        # other error type, or more than a quarter of the canary failing, stops the batch.
        other = [r for r in bad if not str(r.get("error", "")).startswith("stalled")]
        if other or len(bad) > len(head) // 4:
            print(f"canary failed: {len(bad)} of {len(head)} fights with errors; session exits {[r.get('exit') for r in c_rep]}")
            for r in bad[:5]:
                print("  ", r.get("seed"), str(r.get("error"))[:300])
            return 3
        same_c = replay_and_learn(c_res, f"{a.name}.canary")
        print(f"canary ok: {len(head)} fights, no errors, {same_c} identical; {cv.summary(cov, candidates, commit)}", flush=True)
        merged, reports = c_res, c_rep
        remaining -= len(head)
    # Waves: coverage (and so each deck) is updated between waves.
    waves = max(1, a.waves)
    for w in range(waves):
        n = remaining // (waves - w)
        if n <= 0:
            continue
        wave_runs = make_runs(n, len(merged))
        res, rep = play(f"{a.name}.w{w}", wave_runs)
        same_w = replay_and_learn(res, f"{a.name}.w{w}")
        merged += res
        reports += rep
        remaining -= n
        print(f"wave {w}: {n} fights, {same_w} identical; {cv.summary(cov, candidates, commit)}", flush=True)
    out.write_text(json.dumps({"results": merged}), encoding="utf-8")
    report = {"exit": [r.get("exit") for r in reports], "seconds": max((r.get("seconds") or 0) for r in reports),
              "network_connections": sorted({c for r in reports for c in r.get("network_connections", [])}),
              "real_profile_touched": any(r.get("real_profile_touched") for r in reports)}
    with open(emu, "w", encoding="utf-8") as f:
        subprocess.run(["dotnet", str(ROOT / "tools/emudump/bin/Release/net9.0/EmuDump.dll"), "fight", str(out)],
                       stdout=f, check=False)
    cmp = subprocess.run([sys.executable, "-m", "tools.fidelity.fight_compare", str(out), str(emu)],
                         capture_output=True, text=True, cwd=ROOT, check=False).stdout

    t_end = time.time()
    same, groups, errors = 0, collections.defaultdict(list), []
    current = None
    for line in cmp.splitlines():
        if " steps identical" in line:
            same += 1
        elif "error:" in line and not line.startswith(" "):
            errors.append(line[:300])
        elif "first difference at step" in line:
            current = line
        elif line.startswith("    .") and current:
            field = line.split(":")[0].strip()
            card = current.split("'card': '")[1].split("'")[0] if "'card': '" in current else (
                "end_turn" if "end_turn" in current else "start")
            groups[f"{field} after {card}"].append(current[:200])
            current = None
    # Speed metrics. Headline: verified fights per minute = fights played in the game AND
    # replayed and compared in the emulator, over the batch's whole wall clock (start-up,
    # stalls and resumes included).
    played = [r for r in merged if r.get("steps")]
    actions = sum(max(0, len(r["steps"]) - 1) for r in played)
    # Verified transitions: steps where the emulator reproduced the game's next state, i.e.
    # every step before a fight's first divergence. The honest headline (SPEED_PLAN.md).
    emus_all = json.loads(emu.read_text(encoding="utf-8"))["results"] if emu.exists() else []
    vt = 0
    for g, e in zip(merged, emus_all):
        if g.get("steps") and not g.get("error") and not e.get("error"):
            idx = cv.first_divergence(g, e, nm)
            vt += (len(g["steps"]) if idx is None else idx) - 1
    # Re-anchored: every step whose card play or end of turn the emulator reproduced from the
    # game's state before it, divergences earlier in the fight notwithstanding.
    rea = reanchor(out, JOBS / f"{a.name}.reanchor.json")
    rtot = rc.summarise([rc.analyse(g, r, nm, "stop", rc.keys_of(rea)) for g, r in rc.pair(merged, rea["results"])])["totals"]
    vt_first, vt = vt, rtot.get("verified", 0)
    minutes = max(1e-9, (t_end - t_start) / 60)
    per_fight = [r["s_per_fight_per_server"] for r in reports if r.get("s_per_fight_per_server")]
    stalls = sum(1 for r in merged if str(r.get("error", "")).startswith("stalled"))
    metrics = {
        "batch": a.name, "copies": a.parallel, "fights": len(merged), "verified": len(played),
        "minutes": round(minutes, 2), "vt_per_min": round(vt / minutes), "vt_first_div_per_min": round(vt_first / minutes),
        "reanchored": rtot.get("reanchored", 0), "fight_equiv_per_min": round(vt / minutes / 18.9, 1),
        "verified_per_min": round(len(played) / minutes, 1),
        "actions_per_min": round(actions / minutes), "s_per_fight_per_copy": round(sum(per_fight) / max(1, len(per_fight)), 2),
        "identical": same, "identical_pct": round(100 * same / max(1, len(played)), 1),
        "stalls": stalls, "errors": len([r for r in merged if r.get("error")]),
    }
    mfile = JOBS / "metrics.v2.csv"  # v2: re-anchored headline (vt_per_min) plus the first-divergence one
    if not mfile.exists():
        mfile.write_text(",".join(metrics) + chr(10), encoding="utf-8")
    with open(mfile, "a", encoding="utf-8") as f:
        f.write(",".join(str(v) for v in metrics.values()) + chr(10))
    speed = (f"SPEED {metrics['vt_per_min']} verified transitions/min (= {metrics['fight_equiv_per_min']} fight-equivalents/min; "
             f"{metrics['verified_per_min']} fights compared/min, {metrics['actions_per_min']} actions/min), "
             f"{metrics['s_per_fight_per_copy']} s/fight per copy x {a.parallel} copies, {metrics['minutes']} min; "
             f"identical {metrics['identical_pct']}%, stalls {stalls}; "
             f"re-anchored {vt} of {metrics['reanchored']} steps match (first-divergence method: {metrics['vt_first_div_per_min']}/min)")
    lines = [speed, f"batch {a.name}: {a.fights} fights, session exit {report.get('exit')}, {report.get('seconds')} s, "
             f"network {report.get('network_connections')}, real profile touched {report.get('real_profile_touched')}",
             f"identical: {same}", f"diverging: {sum(len(v) for v in groups.values())}", f"errors: {len(errors)}"]
    for field, items in sorted(groups.items(), key=lambda kv: -len(kv[1])):
        lines.append(f"  {field}: {len(items)}  e.g. {items[0]}")
    lines += ["  " + e for e in errors[:10]]
    packs = JOBS / f"{a.name}.packs"  # one context pack per likely bug, plus an index
    pk = subprocess.run([sys.executable, "-m", "tools.fidelity.context_pack", str(out), str(emu), "--out", str(packs)],
                        capture_output=True, text=True, cwd=ROOT, check=False)
    lines.append(f"context packs: {packs / 'index.md'}" if pk.returncode == 0 else f"context packs failed: {pk.stderr[-300:]}")
    text = "\n".join(lines)
    (JOBS / f"{a.name}.summary.txt").write_text(text + "\n\n" + cmp, encoding="utf-8")
    print(text)
    return 0


if __name__ == "__main__":
    sys.exit(main())
