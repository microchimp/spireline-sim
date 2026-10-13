"""Score one emulator commit on the reference set (referee/refset.py): the accuracy numbers to quote.

    python -m tools.fidelity.refset_score [--version v2] [--subset all|pilot] [--emu <emulator checkout>]
        [--procs 4] [--label <name>] [--codex [--codex-emu <checkout>] [--codex-limit N]]
        [--allow-dirty-emu] [--allow-unchecked-record]

Run from the fork's root. --emu is a checkout with EmuDump built (tools/emudump/bin/Release/net9.0); its
commit is what the numbers are for, so a checkout with uncommitted changes is refused (--allow-dirty-emu scores
it anyway, labelled <commit>-dirty and NOT QUOTABLE), and so is a record made before the driver reported its
build (--allow-unchecked-record, also NOT QUOTABLE). Every recorded reference fight is replayed (EmuDump fight) and
re-anchored (EmuDump reanchor) in --procs processes (one emulator per process) and judged with this checkout's
comparers; each emulator result is paired with its reference by position AND seed (a seed mismatch is an error).

The headline line states, per part: specs, kept, excluded (by cause), kept with a realism flag; identical fights
as a share of the kept ones AND counting every exclusion as not identical; steps verified; the emulator commit and
the driver build and capabilities the references were recorded with. Then a table per act x room type (specs,
kept, excluded, flagged, identical both ways), so losses in a cell are visible. Stored per emulator commit in
D:/spireline/referee/refset/<version>/scores/<subset>-<label>.json and appended to scores/history.jsonl.
  realistic   fights identical to the end; steps verified (re-anchored: each recorded step tried from the game's
              own state before it); per act and room type.
  coverage    fights identical; the items not yet clean: card variants with fewer than 5 verified plays or a play
              the emulator gets wrong, encounters with fewer than 3 identical fights, relics whose effect FIRED
              (a relic flash) at fewer than 3 verified steps or at a step the emulator got wrong (a relic merely
              held is not covered), potions used fewer than 3 times verified. Counted over the whole set.
  regression  classes that pass: every fight identical AND the class's trigger reached in one of them (its card
              played, its relic fired, its potion used, its power up when an enemy hit landed); classes about the
              driver itself are counted apart.
  codex runs  (--codex) runs matching to the end; per component, the floors matching among the floors reached
              before each run's first difference (a stop ends the count), stops by class with the profile-dependent
              ones apart, components at 0%, and the runs the walker skipped against the set's own count.
  open bug classes: distinct first-divergence causes (field group after a card, end of turn, start, potion).
"Identical" compares, at every step: the player's HP, max HP, block, energy and powers; hand, draw, discard and
exhaust piles (card id and upgrade); each living enemy's name, HP, block, powers and intent; nine RNG counters;
the potion belt (COMPARED below). It does not compare relic counters, card costs or enchantments, gold or the
hidden counters the comparer skips (NOT_COMPARED).
"""

from __future__ import annotations

import argparse
import collections
import json
import os
import re
import subprocess
import sys
import time
from pathlib import Path

from tools.fidelity import fight_compare as fc
from tools.fidelity import reanchor_compare as rc

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "referee"))
import refset  # noqa: E402  (referee/refset.py: the record's causes and the classes' triggers)

REFSET = Path("D:/spireline/referee/refset")
CODEX = Path("D:/spireline/data/spire-codex/runs-v0.107.1.jsonl.gz")
DECOMPILED = Path("D:/spireline/referee/decompiled")
PARTS = ("realistic", "coverage", "regression")
BELOW_NORMAL = 0x00004000
QUOTA = {"card": 5, "encounter": 3, "relic": 3, "potion": 3}
COMPARED = ("player hp, max hp, block, energy, powers", "hand, draw, discard, exhaust (card id, upgraded)",
            "living enemies: name, hp, block, powers, intent",
            "rng counters: Shuffle, CombatTargets, MonsterAi, CombatCardSelection, CombatCardGeneration, CombatEnergyCosts, "
            "CombatPotionGeneration, CombatOrbGeneration, Niche", "potion belt")
NOT_COMPARED = ("relic counters", "card costs and enchantments", "gold", "hidden power counters the comparer skips")
# What a realistic fight still is not, for every spec (the per-spec flags are the record's): the run's RNG streams
# stand where the seed's run is after Neow, not where the real run had them on that floor; the fight is met on the
# act's row-1 room, not the real run's map coordinate (floor numbers differ); actions come from a chooser, not the
# player who played the run.
REALISM_LIMITS = ("rng streams at the seed's post-Neow state, not the real run's at that floor",
                  "met at the act's row-1 room, not the real map coordinate", "actions from a chooser, not the real player")


def git_commit(path: Path) -> dict:
    def g(*args):
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(g("status", "--porcelain", "--untracked-files=no"))}


def emudump(dll: Path, mode: str, results: list, work: Path, procs: int) -> list:
    """EmuDump <mode> over results in up to procs processes; [meta, results in input order]."""
    if not results:
        return [{}, []]
    k = max(1, min(procs, len(results)))
    size = -(-len(results) // k)
    running = []
    for i in range(0, len(results), size):
        part = results[i:i + size]
        src, dst = work / f"{mode}.p{i // size}.in.json", work / f"{mode}.p{i // size}.json"
        src.write_text(json.dumps({"results": part}), encoding="utf-8")
        flags = BELOW_NORMAL if os.name == "nt" else 0
        out = open(dst, "w", encoding="utf-8")
        running.append((src, dst, len(part), out, subprocess.Popen(["dotnet", str(dll), mode, str(src)], stdout=out,
                                                                   creationflags=flags)))
    got = []
    for src, dst, n, out, p in running:
        p.wait()
        out.close()
        doc = json.loads(dst.read_text(encoding="utf-8")) if dst.stat().st_size else {"results": []}
        rs = doc.get("results", [])
        got.append((doc, rs + [{"error": "emulator produced no result", "steps": []}] * (n - len(rs))))
        src.unlink()
        dst.unlink()
    meta = {k_: v for k_, v in got[0][0].items() if k_ != "results"}
    return [meta, [r for _, rs in got for r in rs]]


def first_divergence(g: dict, e: dict, nm: dict):
    """(step, field group, card label, diff) of the first differing step, or None if the fight is identical."""
    gs, es = g.get("steps", []), e.get("steps", [])
    for i, s in enumerate(gs):
        ev = fc.emu_view(es[i]["state"], nm) if i < len(es) else {}
        d = fc.diff(fc.game_view(s["state"]), ev)
        if d:
            return i, rc.group(d[0][0]), label(s.get("action")), d
    if len(es) > len(gs):
        return len(gs), ".steps", "after the last step", [(".steps", len(gs), len(es))]
    return None


def setup_deck(g: dict, e: dict) -> dict | None:
    """The first setup item after which the emulator's deck (in order) is not the game's: both sides log it
    (referee/driver/RefsetSetup.cs, tools/emudump/RefsetSetup.cs). None when every logged deck matches."""
    el = {x.get("i"): x for x in e.get("setup_log", [])}
    for x in g.get("setup_log", []):
        msg = x.get("msg") or ""
        if " | deck: " not in msg or x.get("i") not in el:
            continue
        gd = msg.split(" | deck: ", 1)[1].split(" | ", 1)[0].split(",")
        em = el[x["i"]].get("msg") or ""
        ed = em.split(" | deck: ", 1)[1].split(" | ", 1)[0].split(",") if " | deck: " in em else []
        if gd != ed:
            pos = next((k for k, (a, b) in enumerate(zip(gd, ed)) if a != b), min(len(gd), len(ed)))
            same = sorted(gd) == sorted(ed)
            return {"i": x["i"], "cmd": x.get("cmd"), "pos": pos, "same_cards": same,
                    "game": ",".join(gd[pos:pos + 3]), "emulator": ",".join(ed[pos:pos + 3])}
    return None


def label(action: dict | None) -> str:
    if action is None:
        return "start"
    k = action.get("kind")
    return "END_TURN" if k == "end_turn" else ("POTION:" + action.get("potion", "?") if k == "potion" else action.get("card", "?"))


def evidence(g: dict, ok_steps: set) -> dict:
    """What the recorded game fight shows happening, each with whether the emulator reproduced that step: card plays
    by variant, potion uses, relic flashes (the game's own "this relic did something"), and the player's powers up
    when an enemy with an attack intent took its turn (Combat.cs end_turn steps)."""
    steps = g.get("steps", [])
    plays, uses, powers_hit = [], [], set()
    for j in range(1, len(steps)):
        act = steps[j].get("action") or {}
        if act.get("kind") == "play" and act.get("played", True):
            hand = steps[j - 1]["state"].get("hand", [])
            hi = act.get("hand_index", -1)
            up = 0 <= hi < len(hand) and bool(hand[hi][1])
            plays.append([act.get("card") + ("+" if up else ""), j in ok_steps])
        elif act.get("kind") == "potion" and act.get("used"):
            uses.append([act.get("potion"), j in ok_steps])
        elif act.get("kind") == "end_turn":
            # The enemies' turn: an enemy that showed an attack intent attacked while the player had these powers.
            st = steps[j - 1]["state"]
            if any(str(i[0] if isinstance(i, list) else i).startswith("Attack")
                   for e in st.get("enemies") or [] if e.get("hp", 0) > 0 for i in e.get("intents") or []):
                powers_hit.update(p[0] if isinstance(p, list) else str(p) for p in (st.get("player") or {}).get("powers") or [])
    fires = [[r, (j in ok_steps) if j > 0 else (0 in ok_steps)] for j, r in g.get("relic_flashes") or []]
    return {"plays": plays, "uses": uses, "fires": fires, "powers_hit": sorted(powers_hit)}


def judge(entry_part: str, g: dict, e: dict, r: dict, nm: dict, keys) -> dict:
    from tools.fidelity.setup_compare import compare  # noqa: PLC0415 (feat/console-setup)
    ref = g.get("refset", {})
    v = {"id": ref.get("id"), "part": entry_part, "seed": g.get("seed"), "tag": g.get("tag"),
         "flags": ref.get("flags", []), "chooser": ref.get("chooser") or g.get("chooser") or "random"}
    if e.get("seed") not in (None, g.get("seed")):
        v.update(identical=False, error=f"paired with the emulator's result for seed {e.get('seed')}", transitions=0, verified=0)
        return v
    if e.get("error") and not e.get("steps"):
        v.update(identical=False, error=str(e["error"])[:200], transitions=max(0, len(g.get("steps", [])) - 1), verified=0)
        return v
    sc = compare(g, e) if g.get("setup") else {"unsupported": [], "first": None}
    v["mirror_gap"] = sc["unsupported"]
    v["setup_first"] = sc["first"]
    v["setup_deck"] = setup_deck(g, e)
    fd = first_divergence(g, e, nm)
    v["identical"] = fd is None and not v["mirror_gap"]
    if fd:
        v["step"], v["field"], v["card"], d = fd
        v["diff"] = [[p, str(x)[:100], str(y)[:100]] for p, x, y in d[:3]]
    an = rc.analyse(g, r, nm, "stop", keys)
    v["transitions"] = an["transitions"]
    # A step is verified when the emulator reproduces it: the plain replay matched it (every step before the
    # first divergence), or, from the game's own state before it, the re-anchored step matched.
    plain = fd[0] if fd else len(g.get("steps", []))
    reanchored = {s["j"] for s in an["steps"] if s["j"] > 0 and s["verdict"] == "match"}
    ok_steps = set(range(1, plain)) | reanchored | ({0} if plain > 0 else set())
    v["verified"] = len(ok_steps - {0})
    v["verified_reanchored"] = len(reanchored)
    v["primary"] = [[s["j"], s["card"], sorted({rc.group(p) for p in s["paths"]})[:2]] for s in an["steps"] if s["primary"] and s["j"] > 0][:6]
    v["primary_steps"] = [s["j"] for s in an["steps"] if s["primary"]]
    v["stop"] = (an.get("stop") or {}).get("why")
    v.update(evidence(g, ok_steps))
    st0 = g.get("steps", [{}])[0].get("state", {}) if g.get("steps") else {}
    v["relics"] = st0.get("relics", [])
    v["encounter"] = (g.get("fight") or {}).get("id")
    return v


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}%" if b else "n/a"


def coverage_report(verdicts: list, targets: dict) -> dict:
    plays, plays_ok, bad_play = collections.Counter(), collections.Counter(), collections.Counter()
    enc, enc_ok = collections.Counter(), collections.Counter()
    fires, fires_ok, bad_fire, held = collections.Counter(), collections.Counter(), collections.Counter(), collections.Counter()
    pot, pot_ok, bad_pot = collections.Counter(), collections.Counter(), collections.Counter()
    for v in verdicts:
        for c, ok in v.get("plays", []):
            plays[c] += 1
            plays_ok[c] += ok
        for j, card, _ in v.get("primary", []):
            if card and card != "END_TURN" and not card.startswith("POTION:"):
                bad_play[card] += 1
            elif card and card.startswith("POTION:"):
                bad_pot[card[7:]] += 1
        for p, ok in v.get("uses", []):
            pot[p] += 1
            pot_ok[p] += ok
        if v.get("encounter"):
            enc[v["encounter"]] += 1
            enc_ok[v["encounter"]] += bool(v.get("identical"))
        for r in set(v.get("relics", [])):
            held[r] += 1
        for r, ok in v.get("fires", []):
            fires[r] += 1
            fires_ok[r] += ok
            if not ok:
                bad_fire[r] += 1
    lists = targets.get("lists", {})
    out = {"not_reached": {}, "not_clean": {}, "counts": {}}

    def judge_items(kind, items, seen, ok, bad=None, base_of=lambda x: x):
        nr = [x for x in items if seen[x] < QUOTA[kind]]
        nc = [x for x in items if seen[x] >= QUOTA[kind] and (ok[x] < QUOTA[kind] or (bad and bad[base_of(x)] > 0))]
        out["not_reached"][kind], out["not_clean"][kind] = nr, nc
        out["counts"][kind] = {"items": len(items), "clean": len(items) - len(nr) - len(nc), "not_reached": len(nr), "not_clean": len(nc)}
    judge_items("card", lists.get("cards", []), plays, plays_ok, bad_play, lambda x: x.rstrip("+"))
    judge_items("encounter", lists.get("encounters", []), enc, enc_ok)
    judge_items("relic", lists.get("relics", []), fires, fires_ok, bad_fire)
    judge_items("potion", lists.get("potions", []), pot, pot_ok, bad_pot)
    out["counts"]["relic"]["held_never_fired"] = sum(1 for r in lists.get("relics", []) if held[r] and not fires[r])
    out["evidence"] = {"plays": dict(plays), "plays_verified": dict(plays_ok), "encounter_fights": dict(enc),
                       "encounter_identical": dict(enc_ok), "relic_fires": dict(fires), "relic_fires_verified": dict(fires_ok),
                       "relic_fights_held": dict(held), "potion_uses": dict(pot), "potion_uses_verified": dict(pot_ok)}
    return out


def trigger_reached(trigger: dict, vs: list) -> bool:
    """Whether one of the class's fights shows its trigger (refset.trigger_of): every kind it names, in one fight."""
    if not trigger or trigger.get("replay"):
        return bool(vs)
    for v in vs:
        played = {c.rstrip("+") for c, _ in v.get("plays", [])}
        fired = {r for r, _ in v.get("fires", [])}
        used = {p for p, _ in v.get("uses", [])}
        if (all(k == "replay" or {"plays": played, "relic_flashes": fired, "potions": used,
                                  "powers_hit": set(v.get("powers_hit", []))}[k] & set(want)
                for k, want in trigger.items())):
            return True
    return False


def regression_report(verdicts: list, entries: dict) -> dict:
    classes = collections.defaultdict(list)
    for v in verdicts:
        if v["part"] == "regression":
            classes[v["class"]].append(v)
    out = {}
    for kind in ("emulator", "driver"):
        names = sorted(c for c, vs in classes.items() if entries.get(vs[0]["id"], {}).get("class_kind", "emulator") == kind)
        passing, untriggered, failing = [], [], []
        for c in names:
            vs = classes[c]
            trig = entries.get(vs[0]["id"], {}).get("trigger") or {}
            if not all(v.get("identical") for v in vs):
                failing.append(c)
            elif trigger_reached(trig, vs):
                passing.append(c)
            else:
                untriggered.append(c)
        out[kind] = {"classes": len(names), "passing": len(passing), "identical_but_trigger_not_reached": untriggered,
                     "failing": failing}
    return out


def codex_score(emu: Path, procs: int, limit: int, work: Path, set_runs: int | None) -> dict:
    dll = emu / "tools" / "emudump" / "bin" / "Release" / "net9.0" / "EmuDump.dll"
    if not (emu / "tools" / "emudump" / "Codex").exists():
        return {"skipped": f"{emu} has no codex replay (feat/codex-replay)"}
    outs, running = [], []
    for i in range(procs):
        out = work / f"codex.s{i}.jsonl"
        args = ["dotnet", str(dll), "codex", str(CODEX), str(out), "--decompiled", str(DECOMPILED), "--character", "IRONCLAD",
                "--ascension", "10", "--shard", f"{i}/{procs}"] + (["--limit", str(-(-limit // procs))] if limit else [])
        running.append(subprocess.Popen(args, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True,
                                        creationflags=BELOW_NORMAL if os.name == "nt" else 0))
        outs.append(out)
    walker_skipped = collections.Counter()
    for p in running:
        err = p.communicate()[1] or ""
        for line in err.splitlines():
            if line.startswith("{") and '"walked"' in line:
                walker_skipped.update(json.loads(line).get("skipped", {}))
    runs, comp = [], collections.defaultdict(lambda: [0, 0])
    stops = collections.Counter()
    for out in outs:
        for line in out.read_text(encoding="utf-8").splitlines():
            r = json.loads(line)
            runs.append(r.get("status"))
            for c, (checked, matched) in r.get("checks", {}).items():
                comp[c][0] += checked
                comp[c][1] += matched
            if r.get("status") == "mismatch":
                s = r.get("stop") or {}
                stops[f"{s.get('class')}: {s.get('component')}"] += 1
            elif r.get("status") == "error":
                stops["error"] += 1
        out.unlink()
    n = len(runs)
    # A walk that stops on a FIGHT-DEPENDENT, PROFILE-DEPENDENT or WALKER difference says nothing against the
    # emulator; runs "judged" are the complete ones and those stopped by an EMULATOR difference.
    emu_stops = sum(k for c, k in stops.items() if c.startswith("EMULATOR"))
    by_class = collections.Counter(c.split(":")[0] for c in stops.elements())
    judged = runs.count("complete") + emu_stops
    walker_ironclad_a10_skips = sum(k for why, k in walker_skipped.items() if "modded" in why)
    return {"runs": n, "complete": runs.count("complete"), "runs_matching_pct": round(100 * runs.count("complete") / n, 1) if n else None,
            "judged": judged, "judged_matching_pct": round(100 * runs.count("complete") / judged, 1) if judged else None,
            "components_note": "floors reached before each run's first difference (a run's count ends at its first stop)",
            "components": {c: {"floors": v[0], "matching": v[1], "pct": round(100 * v[1] / v[0], 1) if v[0] else None}
                           for c, v in sorted(comp.items())},
            "components_at_zero": sorted(c for c, v in comp.items() if v[0] and not v[1]),
            "stops_by_class": dict(by_class), "stops": dict(stops.most_common(20)),
            "runs_in_set": set_runs, "walker_skipped": dict(walker_skipped),
            "gap": (set_runs - n) if set_runs is not None else None, "walker_modded_skips": walker_ironclad_a10_skips,
            "commit": git_commit(emu)}


def runs_score(vdir: Path, emu: Path, work: Path) -> dict:
    """The seeded whole runs (refset.py record --parts runs) through EmuDump run and run_compare2 (feat/run-referee)."""
    latest = vdir / "refs-runs.json"
    if not latest.exists():
        return {"skipped": "no whole runs recorded (refset.py record --parts runs)"}
    if not (emu / "tools" / "emudump" / "RunReplay.cs").exists() or not (ROOT / "tools" / "fidelity" / "run_compare2.py").exists():
        return {"skipped": "the run replay (feat/run-referee) is not in this checkout"}
    refs = Path(json.loads(latest.read_text(encoding="utf-8"))["record"]) / "runs.refs.json"
    dll = emu / "tools" / "emudump" / "bin" / "Release" / "net9.0" / "EmuDump.dll"
    emu_out = work / "runs.emu.json"
    with open(emu_out, "w", encoding="utf-8") as f:
        subprocess.run(["dotnet", str(dll), "run", str(refs)], stdout=f, check=False)
    p = subprocess.run([sys.executable, "-m", "tools.fidelity.run_compare2", "run", str(refs), str(emu_out)], cwd=ROOT,
                       capture_output=True, text=True, check=False, env=dict(os.environ, PYTHONDONTWRITEBYTECODE="1"))
    text = p.stdout + p.stderr
    emu_out.unlink()
    out: dict = {"text": text[-3000:]}
    m = re.search(r"(\d+) runs, (\d+) decisions; (\d+) with no divergence", text)
    if m:
        out.update(runs=int(m[1]), decisions=int(m[2]), runs_matching=int(m[3]))
    firsts = [(int(a), int(b)) for a, b in re.findall(r"first divergence at decision (\d+)/(\d+)", text)]
    if firsts:  # diverging runs: how far into the run the first divergence comes
        out["diverging_runs_decisions_before_first_pct"] = round(100 * sum(a - 1 for a, _ in firsts) / sum(b for _, b in firsts), 1)
    m = re.search(r"fights: (\d+) played, (\d+) identical at every step", text)
    if m:
        out.update(fights=int(m[1]), fights_identical=int(m[2]))
    return out


def block(rows: list, vs: list) -> dict:
    """One part or cell: the record's specs, kept, excluded (by cause) and flagged, and the verdicts of the kept."""
    excl = [r for r in rows if "excluded" in r]
    kept = list(vs)
    ident = sum(1 for v in kept if v.get("identical"))
    clean = [v for v in kept if not v.get("mirror_gap") and not v.get("setup_deck") and not v.get("setup_first")]
    unflagged = [v for v in kept if not v.get("flags")]
    return {"specs": len(rows), "kept": len(kept), "excluded": len(excl),
            "excluded_by_cause": dict(collections.Counter(refset.cause_of(r["excluded"]) for r in excl)),
            "flagged": sum(1 for v in kept if v.get("flags")),
            "identical": ident, "identical_pct_of_kept": round(100 * ident / len(kept), 1) if kept else None,
            "identical_pct_of_specs": round(100 * ident / len(rows), 1) if rows else None,
            "unflagged": len(unflagged), "unflagged_identical": sum(1 for v in unflagged if v.get("identical")),
            "setup_matched": len(clean), "setup_matched_identical": sum(1 for v in clean if v.get("identical")),
            "transitions": sum(v.get("transitions", 0) for v in kept), "verified": sum(v.get("verified", 0) for v in kept),
            "mirror_gaps": sum(1 for v in kept if v.get("mirror_gap")),
            "choosers": dict(collections.Counter(v.get("chooser", "random") for v in kept))}


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--version", default="v2")
    ap.add_argument("--subset", choices=("all", "pilot"), default="all")
    ap.add_argument("--emu", type=Path, default=ROOT, help="emulator checkout with EmuDump built")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--label", default=None, help="a name for this score (default: the commit)")
    ap.add_argument("--codex", action="store_true", help="also walk the codex runs (whole runs, run-level offers)")
    ap.add_argument("--codex-emu", type=Path, default=None, help="checkout with the codex replay (default --emu)")
    ap.add_argument("--codex-limit", type=int, default=0)
    ap.add_argument("--allow-dirty-emu", action="store_true", help="score a checkout with uncommitted changes (NOT QUOTABLE)")
    ap.add_argument("--allow-unchecked-record", action="store_true",
                    help="score a record made before the driver reported its build (NOT QUOTABLE)")
    a = ap.parse_args()
    vdir = REFSET / a.version
    latest = json.loads((vdir / f"refs-{a.subset}.json").read_text(encoding="utf-8"))
    rdir = Path(latest["record"])
    record = json.loads((rdir / "record.json").read_text(encoding="utf-8"))
    manifest = json.loads((vdir / "manifest.json").read_text(encoding="utf-8"))
    entries = {e["id"]: e for e in manifest["specs"]}
    emu = a.emu.resolve()
    dll = emu / "tools" / "emudump" / "bin" / "Release" / "net9.0" / "EmuDump.dll"
    if not dll.exists():
        print(f"no EmuDump build in {emu}: dotnet build -c Release in tools/emudump first", file=sys.stderr)
        return 2
    emu_info = git_commit(emu) | {"emudump_mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(dll.stat().st_mtime))}
    not_quotable = []
    if emu_info["dirty"]:
        if not a.allow_dirty_emu:
            print(f"REFUSED: {emu} has uncommitted changes; its numbers would name no commit (--allow-dirty-emu scores "
                  "it anyway, NOT QUOTABLE)", file=sys.stderr)
            return 2
        not_quotable.append("dirty emulator checkout")
    driver_build = record["stamp"].get("driver_build")
    if not driver_build:
        if not a.allow_unchecked_record:
            print("REFUSED: the record names no driver build (recorded before Capabilities.cs); re-record it, or "
                  "--allow-unchecked-record (NOT QUOTABLE)", file=sys.stderr)
            return 2
        not_quotable.append("record made by an unchecked driver")
    nm = fc.names(json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8")))
    work = vdir / "scores" / "work"
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    verdicts = []
    for part in latest["parts"]:
        p = rdir / f"{part}.refs.json"
        if not p.exists():
            print(f"REFUSED: the record has no {p.name} (parts {latest['parts']})", file=sys.stderr)
            return 2
        games = json.loads(p.read_text(encoding="utf-8"))["results"]
        _, emus = emudump(dll, "fight", games, work, a.procs)
        rmeta, reas = emudump(dll, "reanchor", games, work, a.procs)
        keys = rc.keys_of(rmeta)
        for g, e, r in zip(games, emus, reas):
            verdicts.append(judge(part, g, e, r, nm, keys))
    for v in verdicts:
        en = entries.get(v["id"], {})
        st = en.get("strata", {})
        v["act"], v["room"] = st.get("act"), st.get("room")
        v["class"] = en.get("class")
    rows = record["index"]
    by_part = {p: [r for r in rows if r["part"] == p] for p in PARTS}
    vs_part = {p: [v for v in verdicts if v["part"] == p] for p in PARTS}
    head = {p: block(by_part[p], vs_part[p]) for p in PARTS if by_part[p]}
    table = {}
    for act in (1, 2, 3):
        for room in ("monster", "elite", "boss"):
            rs = [r for r in by_part["realistic"] if (r.get("strata") or entries.get(r["id"], {}).get("strata", {})).get("act") == act
                  and (r.get("strata") or entries.get(r["id"], {}).get("strata", {})).get("room") == room]
            if rs:
                ids = {r["id"] for r in rs}
                table[f"act{act} {room}"] = block(rs, [v for v in vs_part["realistic"] if v["id"] in ids])
    bugs = collections.defaultdict(list)
    for v in verdicts:
        if v.get("mirror_gap"):
            bugs["(setup mirror gap) " + ",".join(map(str, v["mirror_gap"][:2]))].append(v["id"])
        elif not v.get("identical") and v.get("setup_deck"):
            sd = v["setup_deck"]
            bugs[f"(setup) deck {'order' if sd['same_cards'] else 'cards'} after {sd['cmd']}"].append(v["id"])
        elif not v.get("identical"):
            bugs[f"{v.get('field', '(error)')} after {v.get('card', '?')}"].append(v["id"])
    open_bugs = [{"cause": k, "fights": len(ids), "example": ids[0]} for k, ids in sorted(bugs.items(), key=lambda kv: -len(kv[1]))]
    cov = coverage_report(verdicts, manifest.get("coverage_targets", {}))
    reg = regression_report(verdicts, entries)
    codex = (codex_score((a.codex_emu or emu).resolve(), a.procs, a.codex_limit, work,
                         (manifest.get("codex_stats") or {}).get("ironclad_a10_runs")) if a.codex else {"skipped": "not run (--codex)"})
    runs = runs_score(vdir, emu, work)
    commit = emu_info["commit"][:12] + ("-dirty" if emu_info["dirty"] else "")
    name = a.label or commit
    score = {"set": f"{manifest['set']} {manifest['version']}", "subset": a.subset, "emulator": emu_info, "label": name,
             "commit_label": commit, "not_quotable": not_quotable, "comparer": git_commit(ROOT), "record": record["stamp"],
             "record_dir": str(rdir), "scored": time.strftime("%Y-%m-%d %H:%M"), "minutes": round((time.time() - t0) / 60, 1),
             "headline": head, "realistic_table": table, "record_summary": record.get("summary"),
             "flags_by_item": (record.get("summary") or {}).get("flags_by_item"),
             "exclusions": [{"id": r["id"], "part": r["part"], "cause": refset.cause_of(r["excluded"]), "why": r["excluded"]}
                            for r in rows if "excluded" in r],
             "compared": COMPARED, "not_compared": NOT_COMPARED, "realism_limits": REALISM_LIMITS,
             "regression": reg, "coverage": cov, "codex": codex, "runs": runs, "open_bug_classes": open_bugs, "verdicts": verdicts}
    sdir = vdir / "scores"
    hist = sdir / "history.jsonl"
    prev = None
    if hist.exists():
        for line in hist.read_text(encoding="utf-8").splitlines():
            x = json.loads(line)
            if x.get("subset") == a.subset and x.get("label") != name:
                prev = x
    line = summary_line(score)
    (sdir / f"{a.subset}-{name}.json").write_text(json.dumps(score, indent=1), encoding="utf-8")
    with open(hist, "a", encoding="utf-8") as f:
        f.write(json.dumps({"label": name, "subset": a.subset, "commit": emu_info["commit"], "dirty": emu_info["dirty"],
                            "scored": score["scored"], "headline": head, "regression": {k: v["passing"] for k, v in reg.items()},
                            "line": line, "not_quotable": not_quotable,
                            "codex": {k: codex.get(k) for k in ("runs", "runs_matching_pct")}}) + "\n")
    print(line)
    if prev:
        print("  vs " + prev["label"] + ": " + delta(prev["headline"], head))
    print(render_table(score))
    print(f"  stored {sdir / (a.subset + '-' + name + '.json')}")
    return 0


def summary_line(s: dict) -> str:
    h = s["headline"]

    def part(p, steps=False):
        b = h.get(p)
        if not b:
            return f"{p} n/a"
        causes = ", ".join(f"{k} {n}" for k, n in sorted(b["excluded_by_cause"].items(), key=lambda kv: -kv[1]))
        x = (f"{p}: {b['specs']} specs, {b['kept']} kept, {b['excluded']} excluded ({causes or 'none'}), {b['flagged']} flagged; "
             f"identical {b['identical']}/{b['kept']} = {pct(b['identical'], b['kept'])} of kept, "
             f"{pct(b['identical'], b['specs'])} counting exclusions as failures")
        if steps:
            x += f", steps verified {pct(b['verified'], b['transitions'])}"
        return x
    c = s["coverage"]["counts"]
    nc = sum(v["not_clean"] + v["not_reached"] for v in c.values())
    reg = s["regression"]["emulator"]
    cx = s["codex"]
    codex = (f"codex runs {cx['complete']}/{cx['runs']} matching to the end ({cx['runs_matching_pct']}%) @ "
             f"{cx['commit']['commit'][:12]}" if cx.get("runs") else "codex runs n/a")
    rn = s.get("runs") or {}
    runs = (f"whole runs {rn['runs_matching']}/{rn['runs']} matching" if rn.get("runs") else "whole runs n/a")
    db = s["record"].get("driver_build") or {}
    nq = f"NOT QUOTABLE ({'; '.join(s['not_quotable'])}) " if s["not_quotable"] else ""
    return (f"{nq}{s['set']} [{s['subset']}] emulator {s['commit_label']}, references by driver {str(db.get('commit', '?'))[:12]} "
            f"[{','.join(db.get('capabilities', [])) or 'unchecked'}; settle {db.get('settle', '?')}] | "
            f"{part('realistic', True)} | {part('coverage')}, {nc} items not yet clean | "
            f"regression {reg['passing']}/{reg['classes']} emulator classes pass with their trigger reached "
            f"({len(reg['identical_but_trigger_not_reached'])} identical without it) | {codex} | {runs} | "
            f"{len(s['open_bug_classes'])} open bug classes")


def delta(old: dict, new: dict) -> str:
    out = []
    for p in PARTS:
        if p in old and p in new:
            a, b = old[p], new[p]
            ak = a.get("kept", a.get("fights", 0))
            out.append(f"{p} identical {pct(a['identical'], ak)} -> {pct(b['identical'], b['kept'])} of kept, "
                       f"verified {pct(a['verified'], a['transitions'])} -> {pct(b['verified'], b['transitions'])}")
    return "; ".join(out)


def render_table(s: dict) -> str:
    rows = [f"  {'realistic':<14} {'specs':>5} {'kept':>5} {'excl':>5} {'flag':>5} {'identical':>9} {'% kept':>7} "
            f"{'% specs':>7} {'unflagged id.':>13} {'steps':>6} {'verified':>8}"]

    def row(k, b):
        return (f"  {k:<14} {b['specs']:>5} {b['kept']:>5} {b['excluded']:>5} {b['flagged']:>5} {b['identical']:>9} "
                f"{pct(b['identical'], b['kept']):>7} {pct(b['identical'], b['specs']):>7} "
                f"{str(b['unflagged_identical']) + '/' + str(b['unflagged']):>13} {b['transitions']:>6} {pct(b['verified'], b['transitions']):>8}")
    for k, b in s["realistic_table"].items():
        rows.append(row(k, b))
    for p, b in s["headline"].items():
        rows.append(row("= " + p, b) + (f"  ({b['mirror_gaps']} mirror gaps)" if b["mirror_gaps"] else "")
                    + f"  choosers {b['choosers']}")
    for p, b in s["headline"].items():
        if b["excluded_by_cause"]:
            rows.append(f"  {p} excluded by cause: {b['excluded_by_cause']}")
    if s.get("flags_by_item"):
        rows.append("  realism flags (kept fights): " + "; ".join(f"{k} {n}" for k, n in list(s["flags_by_item"].items())[:14]))
    c = s["coverage"]["counts"]
    rows.append("  coverage items (whole set): " + "; ".join(
        f"{k} {v['clean']}/{v['items']} clean ({v['not_reached']} not reached, {v['not_clean']} diverging)" for k, v in c.items())
        + f"; relics held but never fired: {c.get('relic', {}).get('held_never_fired', 0)}")
    rows.append("  open bug classes (first divergence, fights): " + "; ".join(
        f"{b['cause']} {b['fights']}" for b in s["open_bug_classes"][:12]))
    for kind, r in s["regression"].items():
        if r["classes"]:
            rows.append(f"  regression ({kind}): {r['passing']}/{r['classes']} pass with the trigger reached; identical without "
                        f"it: {', '.join(r['identical_but_trigger_not_reached'][:6]) or '-'}; failing: {', '.join(r['failing'][:10]) or '-'}")
    cx = s["codex"]
    if cx.get("components"):
        rows.append("  codex, " + cx["components_note"] + ": " + "; ".join(
            f"{k} {v['pct']}% of {v['floors']}" for k, v in cx["components"].items()))
        rows.append(f"  codex stops by class: {cx['stops_by_class']}; components at 0%: {', '.join(cx['components_at_zero']) or '-'}; "
                    f"runs in the set {cx['runs_in_set']}, walked {cx['runs']}, walker skipped {cx['walker_skipped']}")
    rows.append("  compared: " + "; ".join(s["compared"]) + " | not compared: " + ", ".join(s["not_compared"]))
    rows.append("  realism limits of every realistic fight: " + "; ".join(s["realism_limits"]))
    return "\n".join(rows)


if __name__ == "__main__":
    sys.exit(main())
