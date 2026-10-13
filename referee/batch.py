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

    rng = random.Random(a.start)
    runs = [{"kind": "combat", "seed": "".join(rng.choice(ALPHABET) for _ in range(10)), "character": "IRONCLAD",
             "ascension": 10, "neow": i % 3, "chooser_seed": a.start * 1000 + i, "max_actions": 150,
             "deck_add": rng.sample(candidates, a.deck_cards)}
            for i in range(a.fights)]
    job, out, emu = (JOBS / f"{a.name}{ext}" for ext in (".json", ".out.json", ".emu.json"))
    job.write_text(json.dumps({"out": str(out).replace("\\", "/"), "runs": runs}), encoding="utf-8")

    sys.path.insert(0, str(ROOT / "referee"))
    from run_referee import install  # noqa: PLC0415
    install(Path(GAME), ROOT / "referee" / "driver" / "bin" / "Release" / "net9.0")  # once, before any copy starts

    def play(name: str, subset: list) -> tuple[list, list]:
        """Run `subset` on up to --parallel game copies; results in subset order."""
        k = max(1, min(a.parallel, len(subset)))
        procs = []
        for i in range(k):
            shard = JOBS / f"{name}.s{i}.json"
            shard_out = JOBS / f"{name}.s{i}.out.json"
            shard.write_text(json.dumps({"out": str(shard_out).replace("\\", "/"), "fast_reset": True, "no_preload": True,
                                         "runs": subset[i::k]}), encoding="utf-8")
            procs.append((i, shard_out, launch(shard, shard_out, f"{name}.s{i}")))
        reports, shard_results = [], [[] for _ in range(k)]
        for i, shard_out, p in procs:
            text = p.communicate()[0]
            reports.append(json.loads(text[text.index("{"):]) if "{" in text else {"exit": "no report", "tail": text[-500:]})
            got = json.loads(shard_out.read_text(encoding="utf-8"))["results"] if shard_out.exists() else []
            # Resume: a copy that stalled (watchdog) or died keeps its saved fights; the rest of
            # its share runs again in a fresh session, up to two more times.
            mine = subset[i::k]
            for attempt in range(2):
                if len(got) >= len(mine):
                    break
                # The fight that stalled is recorded, not retried (a hang there is deterministic).
                got.append({"seed": mine[len(got)]["seed"], "steps": [],
                            "error": "stalled: the game made no progress for 20 s during this fight"})
                if len(got) >= len(mine):
                    break
                rest = mine[len(got):]
                rjob, rout = JOBS / f"{name}.s{i}.r{attempt}.json", JOBS / f"{name}.s{i}.r{attempt}.out.json"
                rjob.write_text(json.dumps({"out": str(rout).replace("\\", "/"), "fast_reset": True, "no_preload": True, "runs": rest}),
                                encoding="utf-8")
                rtext = launch(rjob, rout, f"{name}.s{i}.r{attempt}").communicate()[0]
                reports.append(json.loads(rtext[rtext.index("{"):]) if "{" in rtext else {"exit": "no report"})
                got += json.loads(rout.read_text(encoding="utf-8"))["results"] if rout.exists() else []
            shard_results[i] = got
        merged = [None] * len(subset)
        for i, res in enumerate(shard_results):
            for j, r in enumerate(res):
                merged[i + j * k] = r
        merged = [r if r is not None else {"seed": subset[n]["seed"], "error": "shard produced no result", "steps": []}
                  for n, r in enumerate(merged)]
        return merged, reports

    def launch(job: Path, out_path: Path, scratch: str):
        return subprocess.Popen(
            [sys.executable, str(ROOT / "referee" / "run_referee.py"), "--game", GAME, "--job", str(job), "--no-install",
             "--timeout", str(a.timeout), "--scratch", str(Path(GAME).parent / "scratch" / scratch)],
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)

    # Canary: a few fights per copy first, compared before the rest is spent on a broken setup.
    if a.canary and len(runs) > a.canary * a.parallel:
        head = runs[: a.canary * a.parallel]
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
        c_out, c_emu = JOBS / f"{a.name}.canary.out.json", JOBS / f"{a.name}.canary.emu.json"
        c_out.write_text(json.dumps({"results": c_res}), encoding="utf-8")
        with open(c_emu, "w", encoding="utf-8") as f:
            subprocess.run(["dotnet", str(ROOT / "tools/emudump/bin/Release/net9.0/EmuDump.dll"), "fight", str(c_out)],
                           stdout=f, check=False)
        c_cmp = subprocess.run([sys.executable, "-m", "tools.fidelity.fight_compare", str(c_out), str(c_emu)],
                               capture_output=True, text=True, cwd=ROOT, check=False).stdout
        print(f"canary ok: {len(head)} fights, no errors, {c_cmp.count(' steps identical')} identical", flush=True)
        rest, reports = play(a.name, runs[len(head):])
        merged = c_res + rest
        reports = c_rep + reports
    else:
        merged, reports = play(a.name, runs)
    out.write_text(json.dumps({"results": merged}), encoding="utf-8")
    report = {"exit": [r.get("exit") for r in reports], "seconds": max((r.get("seconds") or 0) for r in reports),
              "network_connections": sorted({c for r in reports for c in r.get("network_connections", [])}),
              "real_profile_touched": any(r.get("real_profile_touched") for r in reports)}
    with open(emu, "w", encoding="utf-8") as f:
        subprocess.run(["dotnet", str(ROOT / "tools/emudump/bin/Release/net9.0/EmuDump.dll"), "fight", str(out)],
                       stdout=f, check=False)
    cmp = subprocess.run([sys.executable, "-m", "tools.fidelity.fight_compare", str(out), str(emu)],
                         capture_output=True, text=True, cwd=ROOT, check=False).stdout

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
    lines = [f"batch {a.name}: {a.fights} fights, session exit {report.get('exit')}, {report.get('seconds')} s, "
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
