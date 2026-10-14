#!/usr/bin/env python3
"""Rule probes: ask the real game how a disputed rule works, then check the emulator gives the same answer.

    python referee/probe.py list
    python referee/probe.py run --name <name> [--only P01,P03] [--game D:/spireline/referee/game-probe] [--servers 2]
    python referee/probe.py run --name <name> --no-game     # the emulator again on the recorded games (after an engine change)
    python referee/probe.py judge --name <name>             # judge the files already there

Run from the fork's root. A probe (referee/probes.py) is a setup fight (console setup: relics, potions,
the encounter, powers, cards into a pile, energy) with a scripted action list. Every spec is played in
trace mode, so each step records what the game did in order (combat log entries, hooks, RNG draws).

Per probe, one step is measured. Each hypothesis predicts the measurement from the GAME's state before
that step, so the opening hand does not matter. The game's value says which rule the game follows. The
emulator's value comes from the same step re-anchored (EmuDump reanchor: the emulator is put into the
game's state before the step and given the game's action), so it is the emulator's rule from exactly the
game's situation; the plain replay's value is added when the plain replay still matched the game before
the step. A verdict is never AGREE unless both sides produced a value.

Output in --dir (default D:/spireline/referee/probes, local only): <name>.json (specs), .out.json (game),
.emu.json, .reanchor.json, .report.json, .report.txt. The game runs through serve.py with every safety
check (known build, no network, scratch profile), at most --servers game processes and no hot spare.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(ROOT))

import probes as catalog  # noqa: E402

from tools.fidelity import fight_compare as fc  # noqa: E402
from tools.fidelity import reanchor_compare as rc  # noqa: E402

DIR = Path(r"D:\spireline\referee\probes")
GAME = r"D:\spireline\referee\game-probe"
EXPORT = Path(r"D:\spireline\referee\exports\v0.107.1.json")
EMU_DLL = ROOT / "tools" / "emudump" / "bin" / "Release" / "net9.0" / "EmuDump.dll"


# ------------------------------------------------------------------ one view of a state, both sides


class V:
    """A combat state in one shape for the game's snapshot and the emulator's dump (powers by name, cards by id)."""

    def __init__(self, d: dict):
        self.d = d

    @staticmethod
    def game(raw: dict | None) -> V | None:
        if raw is None:
            return None
        p = raw.get("player") or {}
        view = fc.game_view(raw) if raw.get("in_combat") else {}
        alive = (p.get("hp") or 0) > 0
        return V({
            "in_combat": bool(raw.get("in_combat")),
            # How the fight stands: going on, won (the player alive and the fight over) or lost.
            "outcome": "going" if raw.get("in_combat") and alive else ("won" if alive else "lost"),
            # The game's snapshot keeps the player after the fight ends too.
            "player": {"hp": p.get("hp"), "max_hp": p.get("max_hp"), "block": p.get("block"), "energy": p.get("energy"),
                       "powers": dict(fc.game_powers(p.get("powers", [])))},
            "hand": [c[0] for c in raw.get("hand", [])], "draw": [c[0] for c in raw.get("draw", [])],
            "discard": [c[0] for c in raw.get("discard", [])], "exhaust": [c[0] for c in raw.get("exhaust", [])],
            "enemies": [{**e, "powers": dict(e["powers"])} for e in view.get("enemies", [])],
            "potions": raw.get("potions"),
            "rng": {k: v.get("counter") for k, v in (raw.get("rng") or {}).items() if isinstance(v, dict)},
        })

    @staticmethod
    def emu(raw: dict | None, nm: dict) -> V | None:
        """None without a state. A fight that ended has only its run phase (EmuDump dumps nothing else then):
        "Complete" is a lost run (the probes' fights are on the first floor), any other phase a won fight; the
        re-anchored dump's "Over" says neither.
        """
        if raw is None:
            return None
        if not raw.get("in_combat") or "snapshot" not in raw:
            phase = raw.get("run_phase")
            outcome = None if phase in (None, "Over", "Combat") else ("lost" if phase == "Complete" else "won")
            return V({"in_combat": False, "outcome": outcome, "player": None, "hand": [], "draw": [], "discard": [],
                      "exhaust": [], "enemies": [], "potions": None, "rng": {}})
        v = fc.emu_view(raw, nm)
        p = v["player"]
        return V({
            "in_combat": True, "outcome": "going",
            "player": {**p, "powers": dict(p["powers"])},
            "hand": [c[0] for c in v["hand"]], "draw": [c[0] for c in v["draw"]],
            "discard": [c[0] for c in v["discard"]], "exhaust": [c[0] for c in v["exhaust"]],
            "enemies": [{**e, "powers": dict(e["powers"])} for e in v["enemies"]],
            "potions": v.get("potions"), "rng": dict(v.get("rng") or {}),
        })

    def __getattr__(self, k):
        d = self.__dict__["d"]
        if k in ("hp", "max_hp", "block", "energy"):
            return None if d["player"] is None else d["player"][k]
        if k in d:
            return d[k]
        raise AttributeError(k)

    def power(self, name: str) -> int:
        if self.d["player"] is None:
            raise ValueError("no player state (the fight ended)")
        return int(self.d["player"]["powers"].get(name, 0) or 0)

    def enemy_like(self, e: dict, i: int, pre: V) -> dict | None:
        """The enemy that was living enemy i before the step, while no enemy left the fight in it."""
        if len(self.enemies) != len(pre.enemies) or i >= len(self.enemies) or self.enemies[i]["name"] != e["name"]:
            return None
        return self.enemies[i]


class Ctx:
    """What a hypothesis may need beyond the state: card types from the game's own tables (the export)."""

    def __init__(self, card_types: dict[str, str]):
        self.card_types = card_types

    @staticmethod
    def load(export: Path = EXPORT) -> Ctx:
        types: dict[str, str] = {}
        try:
            ex = json.loads(export.read_text(encoding="utf-8"))["results"][0]
            for pool in ex["card_pools"].values():
                for c in pool.get("all", []):
                    types[c["id"]] = c.get("type", "")
        except (OSError, ValueError, KeyError):
            pass
        return Ctx(types)

    def non_attacks(self, hand: list[str], minus_one: str | None = None) -> int:
        cards = list(hand)
        if minus_one in cards:
            cards.remove(minus_one)
        return sum(1 for c in cards if self.card_types.get(c, "Attack") != "Attack")

    @staticmethod
    def hit_after_block(pre: V, block: int) -> int | None:
        """HP the one attacker's intended blow takes through this much block (its Strength added; one hit each)."""
        att = [e for e in pre.enemies if (e.get("intent") or {}).get("kind") == "ATTACK" and (e["intent"] or {}).get("attack")]
        if len(att) != 1:
            return None
        dmg, hits = att[0]["intent"]["attack"]
        per = dmg + att[0]["powers"].get("STRENGTH", 0)
        return max(0, per * hits - block)


# ------------------------------------------------------------------ judging


def _value(fn, *args):
    try:
        return fn(*args), None
    except Exception as ex:  # noqa: BLE001  (a probe that cannot read a state is reported, never a verdict)
        return None, f"{type(ex).__name__}: {ex}"


def jsonable(x):
    return list(x) if isinstance(x, tuple) else x


def judge_one(p: catalog.Probe, g: dict, r: dict | None, e: dict | None, nm: dict, ctx: Ctx, keys=None) -> dict:
    """One probe's verdict: the game's answer, the emulator's, and whether they agree. ``keys``: the re-anchored
    dump's power keys (reanchor_compare.keys_of), for the whole-state verdict of the measured step.
    """
    out: dict = {"id": p.id, "ledger": p.ledger, "question": p.question, "game_class": p.game_class}
    steps = g.get("steps") or []
    if g.get("error"):
        out["status"] = "game error"
        out["why"] = str(g["error"])[:300]
        return out
    j = p.step(steps)
    if j is None or j <= 0 or j >= len(steps):
        out["status"] = "not reached"
        ill = g.get("script_illegal")
        out["why"] = (f"script stopped at step {ill['step']}: {ill['why']}" if ill else
                      f"the game recorded {len(steps)} steps, the probe measures step {j}")
        return out
    out["step"] = j
    out["action"] = steps[j].get("action")
    pre = V.game(steps[j - 1]["state"])
    post_g = V.game(steps[j]["state"])
    if p.need:
        why, err = _value(p.need, pre, ctx)
        if err or why:
            out["status"] = "not reached"
            out["why"] = err or why
            return out
    gv, gerr = _value(p.measure, pre, post_g, ctx)
    out["game"] = jsonable(gv)
    if gerr or gv is None:
        out["status"] = "probe error"
        out["why"] = gerr or "the game's state does not give the measurement"
        return out

    # The emulator from the game's own state: the re-anchored step j, else the plain replay while it still
    # matched the game before the step. A fight the emulator ended counts only for a probe about how it ends.
    def emu_value(raw):
        post = V.emu(raw, nm)
        if post is not None and not post.in_combat and not p.ends:
            return None, "the emulator's fight ended in this step"
        val, err = _value(p.measure, pre, post, ctx)
        return val, err or (None if val is not None else "no measurement from the emulator's state")

    rsteps = (r or {}).get("steps") or []
    rs = rsteps[j] if j < len(rsteps) else None
    if rs is None:
        ev, note = None, (r or {}).get("error") or "re-anchoring stopped before this step"
    elif rs.get("anchor_error") or rs.get("error"):
        ev, note = None, rs.get("anchor_error") or rs.get("error")
    else:
        ev, note = emu_value(rs.get("state"))
    out["emulator_source"] = "re-anchored" if ev is not None else None
    esteps = (e or {}).get("steps") or []
    if j < len(esteps):
        first = next((k for k in range(j) if fc.diff(fc.game_view(steps[k]["state"]), fc.emu_view(esteps[k]["state"], nm))), None)
        if first is None:
            plain, perr = emu_value(esteps[j]["state"])
            out["plain"] = jsonable(plain) if plain is not None else f"n/a ({perr})"
            if ev is None and plain is not None:
                ev, out["emulator_source"] = plain, "plain replay"
        else:
            out["plain"] = f"n/a (plain replay differs from step {first})"
    if note and out["emulator_source"] != "re-anchored":
        out["emulator_note" if ev is None else "reanchor_note"] = note
    out["emulator"] = jsonable(ev)

    preds = {}
    for h in p.hyps:
        pv, perr = _value(h.predict, pre, ctx)
        preds[h.name] = jsonable(pv) if not perr else f"error: {perr}"
    out["predictions"] = preds
    out["game_rule"] = [h.name for h in p.hyps if preds[h.name] == jsonable(gv)]
    out["emulator_rule"] = [h.name for h in p.hyps if ev is not None and preds[h.name] == jsonable(ev)]
    out["answer"] = next((h.answer for h in p.hyps if h.name in out["game_rule"]), None)
    if ev is None:
        out["status"] = "emulator: no value"
    else:
        out["status"] = "AGREE" if jsonable(ev) == jsonable(gv) else "DIFFER"
    if p.hyps and not out["game_rule"]:
        out["game_rule"] = ["none of the hypotheses"]
    if p.evidence:
        txt, err = _value(p.evidence, steps[j].get("trace") or [])
        out["game_trace"] = err or txt
    if p.note:
        out["note"] = p.note
    if r and r.get("steps"):
        # The whole-state check of the same re-anchored step (what reanchor_compare says), beside the probe's own.
        an = rc.analyse(g, r, nm, "plays", keys)
        st = next((x for x in an["steps"] if x["j"] == j), None)
        if st:
            out["full_state"] = st["verdict"] + (": " + ", ".join(st["paths"][:5]) if st["paths"] else "")
    return out


def judge(probe_list: list, games: list, rea: list, emus: list, nm: dict, ctx: Ctx, keys=None) -> list[dict]:
    by_tag = {(g or {}).get("tag") or (g or {}).get("spec", {}).get("tag"): k for k, g in enumerate(games)}
    out = []
    for i, p in enumerate(probe_list):
        k = by_tag.get("probe:" + p.id, i)
        g = games[k] if k < len(games) and games[k] else {"error": "no game result"}
        out.append(judge_one(p, g, rea[k] if k < len(rea) else None, emus[k] if k < len(emus) else None, nm, ctx, keys))
    return out


def short(x) -> str:
    return json.dumps(x, separators=(",", ":")) if not isinstance(x, str) else x


def table(verdicts: list) -> list[str]:
    lines = []
    for v in verdicts:
        head = f"{v['id']} [{','.join(v['ledger'])}] {v['question']}"
        if v["status"] in ("not reached", "game error", "probe error"):
            lines.append(f"{head}\n    {v['status'].upper()}: {v.get('why')}")
            continue
        g_rule = "/".join(v.get("game_rule") or []) or "-"
        e_rule = "/".join(v.get("emulator_rule") or []) or "-"
        src = f" ({v['emulator_source']})" if v.get("emulator_source") else ""
        extra = [("answer", v.get("answer")), ("game did", v.get("game_trace")), ("whole state", v.get("full_state")),
                 ("note", v.get("note")),
                 ("emulator note", v.get("emulator_note")), ("re-anchor note", v.get("reanchor_note"))]
        lines.append(f"{head}\n    game {short(v['game'])} = {g_rule}; emulator{src} {short(v.get('emulator'))} = {e_rule}"
                     f" -> {v['status']}" + (f"   [plain replay {short(v['plain'])}]" if "plain" in v else "")
                     + "".join(f"\n    {k}: {x}" for k, x in extra if x))
    counts: dict = {}
    for v in verdicts:
        counts[v["status"]] = counts.get(v["status"], 0) + 1
    lines.append("totals: " + ", ".join(f"{k} {n}" for k, n in sorted(counts.items())))
    return lines


# ------------------------------------------------------------------ play and replay


def play(name: str, specs: list, game: str, servers: int, out_dir: Path, timeout: float) -> tuple[list, dict]:
    job, out = out_dir / f"{name}.json", out_dir / f"{name}.out.json"
    run_dir = out_dir / "serve" / name
    job.write_text(json.dumps({"runs": specs}, indent=1), encoding="utf-8")
    p = subprocess.run([sys.executable, str(HERE / "serve.py"), "--game", game, "--specs", str(job),
                        "--servers", str(max(1, min(servers, len(specs), 2))), "--spares", "0",
                        "--out", str(out), "--run-dir", str(run_dir)],
                       stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, timeout=timeout, check=False)
    try:
        rep = json.loads((run_dir / "summary.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        rep = {"tail": p.stdout[-800:]}
    rep["exit"] = p.returncode
    got = json.loads(out.read_text(encoding="utf-8"))["results"] if out.exists() else []
    return got, rep


def emudump(mode: str, src: Path, dst: Path) -> dict:
    """EmuDump fight or reanchor over the recorded games, in one process (one emulator per process)."""
    with Path(dst).open("w", encoding="utf-8") as f:
        subprocess.run(["dotnet", str(EMU_DLL), mode, str(src)], stdout=f, check=False, timeout=1800)
    return load(dst)


def load(path: Path) -> dict:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return {"results": []}


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("cmd", choices=("list", "run", "judge"))
    ap.add_argument("--name", default="probes")
    ap.add_argument("--only", default="", help="comma-separated probe ids")
    ap.add_argument("--game", default=GAME)
    ap.add_argument("--servers", type=int, default=2, help="game processes (at most 2)")
    ap.add_argument("--dir", type=Path, default=DIR)
    ap.add_argument("--no-game", action="store_true", help="replay and judge the recorded games again")
    ap.add_argument("--timeout", type=float, default=3600)
    a = ap.parse_args()

    chosen = [catalog.BY_ID[x] for x in a.only.split(",") if x] if a.only else list(catalog.PROBES)
    if a.cmd == "list":
        for p in chosen:
            print(f"{p.id} [{','.join(p.ledger)}] {p.question}  ({len(p.script)} actions; {p.game_class})")
        return 0

    a.dir.mkdir(parents=True, exist_ok=True)
    out = a.dir / f"{a.name}.out.json"
    rep: dict = {}
    if a.cmd == "run" and not a.no_game:
        games, rep = play(a.name, [p.spec() for p in chosen], a.game, a.servers, a.dir, a.timeout)
        bad = rep.get("network_connections") not in ([], None) or rep.get("real_profile_touched")
        print(f"game: exit {rep.get('exit')}, {len(games)} results, network {rep.get('network_connections')}, "
              f"real profile touched {rep.get('real_profile_touched')}")
        if bad or (rep.get("exit") not in (0, None) and not games):
            print("stopping: the game session failed a safety check or produced nothing", file=sys.stderr)
            return 1
    games = json.loads(out.read_text(encoding="utf-8"))["results"]
    if a.cmd == "run":
        emu_doc = emudump("fight", out, a.dir / f"{a.name}.emu.json")
        rea_doc = emudump("reanchor", out, a.dir / f"{a.name}.reanchor.json")
    else:
        emu_doc, rea_doc = load(a.dir / f"{a.name}.emu.json"), load(a.dir / f"{a.name}.reanchor.json")
    emus, rea = emu_doc.get("results", []), rea_doc.get("results", [])
    nm = fc.names(json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8")))
    # With --only, a later judge of the same name still finds each probe by its tag.
    tags = {(g or {}).get("tag") for g in games}
    if not any(t for t in tags):
        specs = json.loads((a.dir / f"{a.name}.json").read_text(encoding="utf-8"))["runs"]
        for g, s in zip(games, specs):
            if g is not None:
                g["tag"] = s.get("tag")
    present = {(g or {}).get("tag") for g in games}
    chosen = [p for p in chosen if "probe:" + p.id in present] or chosen
    verdicts = judge(chosen, games, rea, emus, nm, Ctx.load(), rc.keys_of(rea_doc))
    lines = table(verdicts)
    commit = subprocess.run(["git", "-C", str(ROOT), "rev-parse", "--short", "HEAD"], capture_output=True, text=True).stdout.strip()
    doc = {"name": a.name, "commit": commit, "safety": {k: rep.get(k) for k in ("network_connections", "real_profile_touched", "exit")},
           "verdicts": verdicts}
    (a.dir / f"{a.name}.report.json").write_text(json.dumps(doc, indent=1), encoding="utf-8")
    (a.dir / f"{a.name}.report.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))
    return 0


if __name__ == "__main__":
    os.environ.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    sys.exit(main())
