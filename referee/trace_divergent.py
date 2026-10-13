#!/usr/bin/env python3
"""Re-run a batch's diverging fights with trace mode on and name the first event that differs.

    python referee/trace_divergent.py <batch out.json> --game <own game copy> [--specs <job.json>]
        [--emu <emu.json>] [--reanchor <reanchor.json>] [--max 10] [--servers 1] [--packs DIR] [--seed S]...

Run from the fork's root. A fight replays identically from its spec, so the costly recording (the
game's combat log, actions, hooks and every RNG draw with its caller, hidden state per step:
referee/driver/Trace.cs) is only paid here, on the fights that diverged, never in a normal batch.

For each diverging fight (its first divergence in the plain replay, else its first primary re-anchored
divergence) the spec is re-run through serve.py with "trace": true and max_actions = that step + 2. The
re-run's steps are checked against the batch's (a fight whose re-run differs before the step is not
reproducible, so its divergence may be the game's, not the emulator's). EmuDump replays it re-anchored
with --trace-to the same step, and tools/fidelity/trace_diff.py aligns the two traces. Writes
<packs>/traces/<seed>.trace.txt (and .json), traces/index.md, and the traced records themselves
(traced.out.json, traced.reanchor.json) next to them; context_pack folds the trace diffs into its packs.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
HERE = Path(__file__).resolve().parent
EMU = ROOT / "tools/emudump/bin/Release/net9.0/EmuDump.dll"
NEW_KEYS = {"trace", "hidden", "move", "intents"}  # keys a traced re-run adds to each step


def emudump(mode: str, src: Path, dst: Path, *extra: str) -> dict:
    with open(dst, "w", encoding="utf-8") as f:
        subprocess.run(["dotnet", str(EMU), mode, str(src), *extra], stdout=f, check=True)
    return json.loads(dst.read_text(encoding="utf-8"))


def strip_new(x):
    if isinstance(x, dict):
        return {k: strip_new(v) for k, v in x.items() if k not in NEW_KEYS}
    if isinstance(x, list):
        return [strip_new(v) for v in x]
    return x


def targets(games: list, emus: list, rea: dict, nm: dict) -> dict:
    """seed -> (index, step, why) for every fight with a divergence worth tracing."""
    sys.path.insert(0, str(ROOT))
    from tools.fidelity import reanchor_compare as rc  # noqa: PLC0415
    from tools.fidelity.coverage import first_divergence  # noqa: PLC0415

    keys = rc.keys_of(rea)
    by_seed = {r.get("seed"): r for r in rea["results"]}
    out = {}
    for i, (g, e) in enumerate(zip(games, emus)):
        if not g.get("steps") or g.get("seed") is None:
            continue
        j = first_divergence(g, e, nm)
        if j is not None and j > 0:
            out[g["seed"]] = (i, j, "first divergence")
            continue
        r = by_seed.get(g["seed"])
        if r is None:
            continue
        a = rc.analyse(g, r, nm, "stop", keys)
        prim = [s["j"] for s in a["steps"] if s["j"] > 0 and s["verdict"] == "diverge" and s["primary"]]
        if prim:
            out[g["seed"]] = (i, prim[0], "primary re-anchored divergence")
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("out_json", type=Path)
    ap.add_argument("--game", required=True, help="your own game copy (referee/linkcopy.py), never the shared one")
    ap.add_argument("--specs", type=Path, help="the batch's job file (default: <name>.json beside <name>.out.json)")
    ap.add_argument("--emu", type=Path, help="EmuDump fight output for the batch (default: computed)")
    ap.add_argument("--reanchor", type=Path, help="EmuDump reanchor output for the batch (default: computed)")
    ap.add_argument("--packs", type=Path, help="context pack directory (default: <name>.packs beside the out file)")
    ap.add_argument("--max", type=int, default=10, help="fights to trace")
    ap.add_argument("--seed", action="append", default=[], help="trace only these seeds")
    ap.add_argument("--servers", type=int, default=1, choices=(1, 2, 3))
    ap.add_argument("--stall-s", type=int, default=120)
    ap.add_argument("--reuse", action="store_true", help="keep the traced game records of an identical earlier job; redo the emulator side")
    a = ap.parse_args()

    sys.path.insert(0, str(ROOT))
    from tools.fidelity import trace_diff  # noqa: PLC0415
    from tools.fidelity.fight_compare import names  # noqa: PLC0415

    nm = names(json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8")))
    name = a.out_json.name.removesuffix(".out.json")
    specs_path = a.specs or a.out_json.with_name(name + ".json")
    specs = json.loads(specs_path.read_text(encoding="utf-8"))["runs"]
    games = json.loads(a.out_json.read_text(encoding="utf-8"))["results"]
    if len(specs) != len(games):
        print(f"{specs_path} has {len(specs)} specs for {len(games)} results; pass the batch's own job file", file=sys.stderr)
        return 2
    packs = a.packs or a.out_json.with_name(name + ".packs")
    tdir = packs / "traces"
    tdir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    emus = json.loads(a.emu.read_text(encoding="utf-8"))["results"] if a.emu else emudump("fight", a.out_json, tdir / "batch.emu.json")["results"]
    rea = json.loads(a.reanchor.read_text(encoding="utf-8")) if a.reanchor else emudump("reanchor", a.out_json, tdir / "batch.reanchor.json")
    found = targets(games, emus, rea, nm)
    chosen = [(s, v) for s, v in found.items() if not a.seed or s in a.seed][: a.max]
    if not chosen:
        print("no diverging fight to trace")
        return 0
    print(f"{len(found)} diverging fights; tracing {len(chosen)}")

    # The re-run: each spec as it was, traced, stopped two steps after its divergence.
    job = []
    for seed, (i, j, _) in chosen:
        s = dict(specs[i])
        if str(s.get("seed", "")).upper().replace("O", "0").replace("I", "1").strip() != seed:  # SeedHelper.CanonicalizeSeed
            print(f"{seed}: spec {i} is for {s.get('seed')}; pass the batch's own job file", file=sys.stderr)
            return 2
        s["trace"], s["max_actions"] = True, j + 2
        job.append(s)
    jpath = tdir / "traced.json"
    traced_path = tdir / "traced.out.json"
    t1 = time.time()
    same_job = jpath.is_file() and json.loads(jpath.read_text(encoding="utf-8")) == {"runs": job}
    if a.reuse and same_job and traced_path.is_file():
        print("reusing the traced game records of the same job")
    else:
        jpath.write_text(json.dumps({"runs": job}), encoding="utf-8")
        rc_ = subprocess.run([sys.executable, str(HERE / "serve.py"), "--game", a.game, "--specs", str(jpath), "--servers", str(a.servers),
                              "--spares", "0", "--stall-s", str(a.stall_s), "--out", str(traced_path)],
                             capture_output=True, text=True, env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
        summary = json.loads(rc_.stdout[rc_.stdout.index("{\n"):]) if "{\n" in rc_.stdout else {}
        if rc_.returncode != 0 or summary.get("network_connections") or summary.get("real_profile_touched"):
            print("serve.py failed or a safety check did not hold:", rc_.stdout[-1500:], rc_.stderr[-800:], file=sys.stderr)
            return 1
    game_s = time.time() - t1
    traced = json.loads(traced_path.read_text(encoding="utf-8"))["results"]

    # Reproducible? The re-run must give the batch's steps up to the traced step.
    for (_seed, (i, j, _)), t in zip(chosen, traced, strict=True):
        upto = j + 1
        t["_reproducible"] = strip_new(t.get("steps", [])[:upto]) == strip_new(games[i].get("steps", [])[:upto])
    t2 = time.time()
    rea_t = emudump("reanchor", traced_path, tdir / "traced.reanchor.json", "--trace-to", str(max(j for _, (_, j, _) in chosen) + 1))
    emu_s = time.time() - t2
    rmap = {r.get("seed"): r for r in rea_t["results"]}
    index = [f"# Trace diffs for {name}", "", f"{len(chosen)} fights traced: game {game_s:.0f} s, emulator {emu_s:.0f} s.", ""]
    for (seed, (_i, j, why)), t in zip(chosen, traced, strict=True):
        r = rmap.get(seed)
        if t.get("seed") != seed:
            d = {"seed": seed, "step": j, "error": f"the re-run's result is for {t.get('seed')}"}
        elif r is None or not t.get("steps"):
            d = {"seed": seed, "step": j, "error": t.get("error") or "no traced result"}
        elif j >= len(t["steps"]):
            d = {"seed": seed, "step": j, "error": f"the re-run ended at step {len(t['steps']) - 1}: " + str(t.get("error"))[:200]}
        else:
            d = trace_diff.step_diff(t, r, j, nm, trace_diff.keys_of(rea_t))
        d["why"], d["reproducible"] = why, t.get("_reproducible")
        body = trace_diff.text(d)
        if not d["reproducible"]:
            body += "  NOT REPRODUCIBLE: the re-run differs from the batch before this step (a game-side flake?)\n"
        (tdir / f"{seed}.trace.txt").write_text(body, encoding="utf-8")
        (tdir / f"{seed}.trace.json").write_text(json.dumps(d), encoding="utf-8")
        index.append(f"- {seed} step {j} ({why}){'' if d['reproducible'] else ' NOT REPRODUCIBLE'}: "
                     + (d.get("first_difference") or d.get("error") or "no mark differs")[:200])
        sys.stdout.write(body)
    (tdir / "index.md").write_text("\n".join(index) + "\n", encoding="utf-8")
    print(f"\ntraces: {tdir}  (wall {time.time() - t0:.0f} s; game {game_s:.0f} s for {len(chosen)} fights)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
