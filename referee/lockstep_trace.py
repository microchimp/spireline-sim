#!/usr/bin/env python3
"""From a lockstep divergence record to a traced real fight, and the first segment where the game and the emulator differ.

    python referee/lockstep_trace.py --block <agent block dir> --seed <run seed> --decision <lockstep decision> --game <own game copy>
        [--work DIR] [--phase replay|trace|all] [--mode auto|script|natural] [--config a3-v1] [--gen-dir DIR]
        [--emudump EmuDump.dll] [--referee-dir D:/spireline/referee] [--no-lock-check]

Run from the fork's root with the emulator venv (STS2_LIB_PATH = the emulator's build directory). The block directory holds
divergences.jsonl (referee/lockstep.py records: `decision` counts every asked decision of the run, `agent_seed` is [base, run index],
`acted` is the step before it, `diff` is [field, game, emulator]).

1. replay: the record's run is played again in the game copy with the same agent (same agent seed, search config and macro
   generation) under lockstep.play_run, the driver writing the game's run record (reset key "record": the save after every RunPlay
   decision, fights with every step). Every lockstep ask is logged (trail.json: kind, floor, round, enemy hp, the answer). From the
   target decision on no search runs: the referee-only policy plays the rest of the run so the record gets written. The replay is
   checked against the record (floor, round, enemy hp at the target); a path that has moved (the emulator changed since the
   block) is reported and stops the tool.
2. map: `map_decision` takes the lockstep decision to (RunPlay combat decision, fight step).
3. trace: `run_repro.fight_spec` turns the record at that fight into a setup fight (the save before the fight: deck, relics, potions,
   hp, act; not the RNG streams, the floor, or the run history). `script` mode plays the recorded plays up to the step; `natural`
   lets the heuristic chooser play the fight; `auto` tries script, then natural when the script is illegal or nothing differs.
   Each is run plain, then through trace_divergent.py (traced game re-run, EmuDump reanchor --trace-to, tools.fidelity.trace_diff).
   The first differing segment is printed and saved to <work>/<mode>/trace.txt.

Game processes (one per phase and per traced run) only run while QUIET.lock and lab/PAUSE exist in --referee-dir (--no-lock-check
skips the test). The caller makes and removes them.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from unittest import mock

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
for p in (str(HERE), str(ROOT)):
    if p not in sys.path:
        sys.path.insert(0, p)

import lockstep  # noqa: E402
import run_repro  # noqa: E402

REFEREE_DIR = "D:/spireline/referee"


class UsageError(Exception):
    pass


def find_record(block: Path, seed: str, decision: int) -> dict:
    """The block's divergence record for (run seed, lockstep decision)."""
    path = block / "divergences.jsonl"
    if not path.is_file():
        raise UsageError(f"{path} not found (--block is an agent block folder)")
    for line in path.read_text(encoding="utf-8").splitlines():
        if not line.strip():
            continue
        r = json.loads(line)
        if r.get("seed") == seed and int(r.get("decision", -1)) == decision:
            return r
    raise UsageError(f"no record for seed {seed} decision {decision} in {path}")


class TraceLockstep(lockstep.Lockstep):
    """Lockstep that logs every ask (the trail) and, from `stop_after` on, answers by the referee-only policy without the emulator
    (the decisions up to and including `stop_after` are the unchanged Lockstep.decide)."""

    def __init__(self, *a, stop_after: int | None = None, **k):
        super().__init__(*a, **k)
        self.stop_after, self.trail = stop_after, []

    def decide(self, msg: dict) -> int:
        obs = msg.get("obs") or {}
        c = obs.get("combat") or {}
        ent = {"n": self.n, "kind": msg.get("kind"), "floor": obs.get("floor"), "round": c.get("round"),
               "hp": [e.get("hp", e.get("current_hp")) for e in c.get("enemies") or [] if isinstance(e, dict)]}
        self.trail.append(ent)
        if self.stop_after is not None and self.n > self.stop_after:
            self.n += 1
            choice = lockstep.referee_only_policy(msg)
        else:
            choice = super().decide(msg)
        acts = msg.get("actions") or []
        ent["chosen"] = acts[choice] if isinstance(choice, int) and 0 <= choice < len(acts) else None
        return choice


def positive(hp: list) -> list:
    return sorted(h for h in hp if isinstance(h, int) and h > 0)


def map_decision(trail: list, run: dict, n: int) -> dict:
    """The lockstep ask n as a fight of the run record and a step in it.

    The fight is the record's "combat" decision of the ask's floor (one fight per floor). A fight step is one combat ask's answer
    (a lone legal action, a bare End Turn, is taken without an ask but is still a step), so the ask sees the state after
    S = (combat asks of this fight before it) + (unasked steps) actions; S is the first step at or after the ask count whose round
    and living enemy hp are the ask's. Step S is the action the record's `acted` names, the one whose result differs.
    Returns {"fight": decision index, "step": S, "asked_before": k, "verified": bool, "floor", "round"}."""
    ask = next((t for t in trail if t["n"] == n), None)
    if ask is None:
        raise UsageError(f"the replay has no decision {n} (it asked {len(trail)})")
    if ask["kind"] not in lockstep.COMBAT_KINDS:
        raise UsageError(f"decision {n} is a {ask['kind']} ask, not a fight ask")
    fights = [i for i, d in enumerate(run.get("decisions") or []) if d.get("screen") == "combat" and d.get("floor") == ask["floor"]]
    if len(fights) != 1:
        raise UsageError(f"{len(fights)} fights at floor {ask['floor']} in the run record")
    i = fights[0]
    steps = run["decisions"][i]["fight"]["steps"]
    k = sum(1 for t in trail if t["n"] < n and t["floor"] == ask["floor"] and t["kind"] == "combat")
    for s in range(k, len(steps)):
        st = steps[s].get("state") or {}
        if st.get("round") == ask["round"] and positive([e.get("hp") for e in st.get("enemies") or []]) == positive(ask["hp"]):
            return {"fight": i, "step": s, "asked_before": k, "verified": True, "floor": ask["floor"], "round": ask["round"]}
    return {"fight": i, "step": min(k, len(steps) - 1), "asked_before": k, "verified": False, "floor": ask["floor"], "round": ask["round"]}


def reproduced(trail: list, rec: dict) -> tuple[bool, str]:
    """Does the replay's ask at the record's decision look like the record's (floor, round, the game's enemy hp)?"""
    ask = next((t for t in trail if t["n"] == rec["decision"]), None)
    if ask is None:
        return False, f"the replay ended before decision {rec['decision']} ({len(trail)} asks)"
    want_hp = next((d[1] for d in rec.get("diff") or [] if d and d[0] == ".combat.enemy_hp"), None)
    bad = []
    if ask["floor"] != rec.get("floor"):
        bad.append(f"floor {ask['floor']} != {rec.get('floor')}")
    if rec.get("turn") is not None and ask["round"] != rec["turn"]:
        bad.append(f"round {ask['round']} != {rec['turn']}")
    if want_hp is not None and positive(ask["hp"]) != positive(want_hp):
        bad.append(f"enemy hp {ask['hp']} != {want_hp}")
    return not bad, "; ".join(bad) or "floor, round and the game's enemy hp match the record"


def build_spec(run: dict, mapping: dict, rec: dict, mode: str) -> dict:
    """The traced setup fight for the mapped step: run_repro.fight_spec at the fight, with the record's script or none."""
    field = next((d[0] for d in rec.get("diff") or [] if d), ".enemies")
    verdict = {"first": {"component": run_repro.FIGHT, "decision": mapping["fight"], "diffs": [[f"step {mapping['step']} after {{}}: {field}"]]}}
    spec, why = run_repro.fight_spec(run, verdict)
    if spec is None:
        raise UsageError(why)
    step = mapping["step"]
    spec["tag"] += f":lockstep:{rec['decision']}"
    spec["trace"] = True
    if mode == "script":
        steps = run["decisions"][mapping["fight"]]["fight"]["steps"]
        spec["script"] = [steps[j]["action"] for j in range(1, step + 1)]
        spec["max_actions"] = step
    else:
        spec["max_actions"] = max(step + 2, 20)
    return spec


class RecEnv:
    """envproto._Timed that also asks the driver for the run record (reset key "record")."""

    def __init__(self, env, timeout: float, record: Path):
        import envproto  # noqa: PLC0415
        self._t, self.record = envproto._Timed(env, timeout), record

    def reset(self, *a, **k):
        return self._t.reset(*a, record=self.record, **k)

    def step(self, a):
        return self._t.step(a)


def replay(a, rec: dict, work: Path) -> dict:
    """Phase 1: the run again under the same agent, recording the game's run state. Writes run.json, trail.json, replay.json."""
    import envproto  # noqa: PLC0415
    import run_referee as rr  # noqa: PLC0415
    lockstep._paths()
    envproto.below_normal_self()
    game = Path(a.game).resolve()
    rr.check_build(game)
    rr.install(game, HERE / "driver" / "bin" / "Release" / "net9.0")
    if a.gen_dir:
        os.environ["SPIRELINE_GEN_DIR"] = str(a.gen_dir)
    holder: list = []

    def make(*args, **kw):
        ls = TraceLockstep(*args, stop_after=rec["decision"], **kw)
        holder.append(ls)
        return ls

    env = envproto.GameEnv(game, work / "scratch", install=False)
    t0, err, nets = time.time(), None, []
    try:
        with mock.patch.object(lockstep, "Lockstep", make):
            lockstep.play_run(RecEnv(env, a.step_timeout, work / "run.json"), rec["seed"], ascension=a.ascension, sims=a.config,
                              agent_seed=rec["agent_seed"])
    except Exception as e:  # noqa: BLE001 -- the trail so far is still the answer to "did it get there"
        err = f"{type(e).__name__}: {e}"[:300]
    finally:
        try:
            nets = sorted(rr.connections(env.pid))
        finally:
            env.close()
    trail = holder[0].trail if holder else []
    ok, why = reproduced(trail, rec)
    info = {"seed": rec["seed"], "decision": rec["decision"], "asks": len(trail), "reproduced": ok, "why": why, "error": err,
            "network_connections": nets, "record_written": (work / "run.json").is_file(), "wall_s": round(time.time() - t0)}
    (work / "trail.json").write_text(json.dumps(trail), encoding="utf-8")
    (work / "replay.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    return info


def run_serve(game: str, job: Path, out: Path) -> dict:
    r = subprocess.run([sys.executable, str(HERE / "serve.py"), "--game", str(game), "--specs", str(job), "--servers", "1", "--spares", "0",
                        "--stall-s", "120", "--out", str(out)], capture_output=True, text=True,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    summary = json.loads(r.stdout[r.stdout.index("{\n"):]) if "{\n" in r.stdout else {}
    if r.returncode != 0 or summary.get("network_connections") or summary.get("real_profile_touched"):
        raise UsageError("serve.py failed or a safety check did not hold: " + r.stdout[-800:] + r.stderr[-400:])
    return summary


def trace_mode(a, spec: dict, mode: str, work: Path) -> dict:
    """Phase 3 for one mode: the plain fight, then trace_divergent on it. Returns {"mode", "legal", "traced", "text"}."""
    d = work / mode
    d.mkdir(parents=True, exist_ok=True)
    job = d / "fight.json"
    plain = {k: v for k, v in spec.items() if k != "trace"}
    job.write_text(json.dumps({"runs": [plain]}), encoding="utf-8")
    run_serve(a.game, job, d / "fight.out.json")
    res = json.loads((d / "fight.out.json").read_text(encoding="utf-8"))["results"][0]
    illegal = res.get("script_illegal")
    if illegal:
        return {"mode": mode, "legal": False, "traced": False, "text": f"script illegal at step {illegal.get('step')}: {str(illegal.get('why'))[:200]}"}
    env = {**os.environ, "EMUDUMP_DLL": str(a.emudump)} if a.emudump else dict(os.environ)
    r = subprocess.run([sys.executable, str(HERE / "trace_divergent.py"), str(d / "fight.out.json"), "--game", str(a.game), "--servers", "1"],
                       capture_output=True, text=True, env=env)
    text = r.stdout + (("\n" + r.stderr[-800:]) if r.returncode else "")
    (d / "trace.txt").write_text(text, encoding="utf-8")
    return {"mode": mode, "legal": True, "traced": "no diverging fight to trace" not in r.stdout and r.returncode == 0, "text": text}


def locks_ok(a) -> bool:
    ref = Path(a.referee_dir)
    return a.no_lock_check or ((ref / "QUIET.lock").is_file() and (ref / "lab" / "PAUSE").is_file())


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--block", type=Path, required=True, help="the agent block folder (has divergences.jsonl)")
    ap.add_argument("--seed", required=True)
    ap.add_argument("--decision", type=int, required=True)
    ap.add_argument("--game", type=Path, required=True, help="your own game copy")
    ap.add_argument("--work", type=Path, default=None, help="scratch folder (default: ./lockstep-trace-<seed>-<decision>)")
    ap.add_argument("--phase", choices=("replay", "trace", "all"), default="all")
    ap.add_argument("--mode", choices=("auto", "script", "natural"), default="auto")
    ap.add_argument("--config", default="a3-v1", help="the block's search config")
    ap.add_argument("--gen-dir", default=None, help="the block's generation folder (none for generation 0)")
    ap.add_argument("--ascension", type=int, default=10)
    ap.add_argument("--step-timeout", type=float, default=180)
    ap.add_argument("--emudump", type=Path, default=None, help="EmuDump.dll (default: this tree's build)")
    ap.add_argument("--referee-dir", default=REFEREE_DIR)
    ap.add_argument("--no-lock-check", action="store_true")
    return ap


def main(argv: list[str] | None = None) -> int:
    a = parser().parse_args(argv)
    try:
        rec = find_record(a.block, a.seed, a.decision)
        work = a.work or Path(f"lockstep-trace-{a.seed}-{a.decision}")
        work.mkdir(parents=True, exist_ok=True)
        if not locks_ok(a):
            raise UsageError(f"QUIET.lock and lab/PAUSE are not both in {a.referee_dir}: take them first (or --no-lock-check)")
        if a.phase in ("replay", "all"):
            info = replay(a, rec, work)
            print("replay:", json.dumps(info))
            if info["network_connections"] or not info["record_written"] or not info["reproduced"]:
                print("stop: " + (info["why"] if not info["reproduced"] else "no run record or a network connection"))
                return 3
        if not (work / "run.json").is_file() or not (work / "trail.json").is_file():
            raise UsageError(f"no run.json and trail.json in {work}: run --phase replay first")
        run = json.loads((work / "run.json").read_text(encoding="utf-8"))
        run.setdefault("seed", a.seed)
        trail = json.loads((work / "trail.json").read_text(encoding="utf-8"))
        mapping = map_decision(trail, run, a.decision)
        print("map:", json.dumps(mapping))
        if a.phase == "replay":
            return 0
        modes = ["script", "natural"] if a.mode == "auto" else [a.mode]
        for mode in modes:
            res = trace_mode(a, build_spec(run, mapping, rec, mode), mode, work)
            print(f"--- {mode}: " + ("traced" if res["traced"] else "legal, nothing differs" if res["legal"] else "not legal"))
            print(res["text"])
            if res["traced"]:
                return 0
        return 4
    except UsageError as e:
        print(f"lockstep_trace: {e}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())
