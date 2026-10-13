"""Score one emulator commit on the reference set (referee/refset.py): the accuracy numbers to quote.

    python -m tools.fidelity.refset_score [--version v1] [--subset all|pilot] [--emu <emulator checkout>]
        [--procs 4] [--label <name>] [--codex [--codex-emu <checkout>] [--codex-limit N]]

Run from the fork's root. --emu is a checkout with EmuDump built (tools/emudump/bin/Release/net9.0); its
commit is what the numbers are for. Every recorded reference fight is replayed (EmuDump fight) and re-anchored
(EmuDump reanchor) in --procs processes (one emulator per process) and judged with this checkout's comparers.

Headline numbers (one line, then a table), stored per emulator commit in
D:/spireline/referee/refset/<version>/scores/<commit>.json and appended to scores/history.jsonl, with the
change against the previous score of the same subset:
  realistic   fights identical to the end, %; steps verified, % (re-anchored: each recorded step tried from
              the game's own state); both per act and room type.
  coverage    fights identical, %; the items not yet clean: card variants with fewer than 5 verified plays or a
              play the emulator gets wrong, encounters, relics and potions with fewer than 3 identical fights
              (uses, for potions) or one that diverges on them. Counted over the whole set, not only coverage.
  regression  classes whose fights all pass (identical), of all classes.
  codex runs  (--codex) floors matching per component, %; runs matching to the end, %.
  whole runs  the seeded run-referee runs (when recorded and the checkout has the run replay): runs with no
              divergence, decisions before the first one, their fights identical.
  open bug classes: distinct first-divergence causes (field group after a card, end of turn, start, potion).
Fights whose setup the emulator cannot mirror (an "unsupported" item) count as not identical and are listed
apart, as mirror gaps.
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
REFSET = Path("D:/spireline/referee/refset")
CODEX = Path("D:/spireline/data/spire-codex/runs-v0.107.1.jsonl.gz")
DECOMPILED = Path("D:/spireline/referee/decompiled")
PARTS = ("realistic", "coverage", "regression")
BELOW_NORMAL = 0x00004000
QUOTA = {"card": 5, "encounter": 3, "relic": 3, "potion": 3}


def git_commit(path: Path) -> dict:
    def g(*args):
        return subprocess.run(["git", "-C", str(path), *args], capture_output=True, text=True, check=False).stdout.strip()
    return {"commit": g("rev-parse", "HEAD"), "branch": g("rev-parse", "--abbrev-ref", "HEAD"),
            "dirty": bool(g("status", "--porcelain", "--untracked-files=no"))}


def emudump(dll: Path, mode: str, results: list, work: Path, procs: int) -> list:
    """EmuDump <mode> over results in up to procs processes; results in input order."""
    if not results:
        return []
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
        gd = msg.split(" | deck: ", 1)[1].split(",")
        em = el[x["i"]].get("msg") or ""
        ed = em.split(" | deck: ", 1)[1].split(",") if " | deck: " in em else []
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


def judge(entry_part: str, g: dict, e: dict, r: dict, nm: dict, keys) -> dict:
    from tools.fidelity.setup_compare import compare  # noqa: PLC0415 (feat/console-setup)
    ref = g.get("refset", {})
    v = {"id": ref.get("id"), "part": entry_part, "seed": g.get("seed"), "tag": g.get("tag")}
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
    ok_steps = set(range(1, plain)) | reanchored
    v["verified"] = len(ok_steps)
    v["verified_reanchored"] = len(reanchored)
    v["primary"] = [[s["j"], s["card"], sorted({rc.group(p) for p in s["paths"]})[:2]] for s in an["steps"] if s["primary"] and s["j"] > 0][:6]
    v["stop"] = (an.get("stop") or {}).get("why")
    # Coverage evidence: plays by card variant (upgraded from the hand before the play), potion uses.
    steps = g.get("steps", [])
    plays, uses = [], []
    for j in range(1, len(steps)):
        act = steps[j].get("action") or {}
        if act.get("kind") == "play" and act.get("played", True):
            hand = steps[j - 1]["state"].get("hand", [])
            hi = act.get("hand_index", -1)
            up = 0 <= hi < len(hand) and bool(hand[hi][1])
            plays.append([act.get("card") + ("+" if up else ""), j in ok_steps])
        elif act.get("kind") == "potion" and act.get("used"):
            uses.append([act.get("potion"), j in ok_steps])
    v["plays"], v["uses"] = plays, uses
    st0 = steps[0]["state"] if steps else {}
    v["relics"] = st0.get("relics", [])
    v["encounter"] = (g.get("fight") or {}).get("id")
    return v


def pct(a: int, b: int) -> str:
    return f"{100 * a / b:.1f}%" if b else "n/a"


def coverage_report(verdicts: list, targets: dict) -> dict:
    plays, plays_ok, bad_play = collections.Counter(), collections.Counter(), collections.Counter()
    enc, enc_ok = collections.Counter(), collections.Counter()
    rel, rel_ok = collections.Counter(), collections.Counter()
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
            rel[r] += 1
            rel_ok[r] += bool(v.get("identical"))
    lists = targets.get("lists", {})
    out = {"not_reached": {}, "not_clean": {}, "counts": {}}

    def judge_items(kind, items, seen, ok, bad=None, base_of=lambda x: x):
        nr = [x for x in items if seen[x] < QUOTA[kind]]
        nc = [x for x in items if seen[x] >= QUOTA[kind] and (ok[x] < QUOTA[kind] or (bad and bad[base_of(x)] > 0))]
        out["not_reached"][kind], out["not_clean"][kind] = nr, nc
        out["counts"][kind] = {"items": len(items), "clean": len(items) - len(nr) - len(nc), "not_reached": len(nr), "not_clean": len(nc)}
    judge_items("card", lists.get("cards", []), plays, plays_ok, bad_play, lambda x: x.rstrip("+"))
    judge_items("encounter", lists.get("encounters", []), enc, enc_ok)
    judge_items("relic", lists.get("relics", []), rel, rel_ok)
    judge_items("potion", lists.get("potions", []), pot, pot_ok, bad_pot)
    out["evidence"] = {"plays": dict(plays), "plays_verified": dict(plays_ok), "encounter_fights": dict(enc),
                       "encounter_identical": dict(enc_ok), "relic_fights": dict(rel), "relic_identical": dict(rel_ok),
                       "potion_uses": dict(pot), "potion_uses_verified": dict(pot_ok)}
    return out


def codex_score(emu: Path, procs: int, limit: int, work: Path) -> dict:
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
    for p in running:
        p.communicate()
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
    judged = runs.count("complete") + emu_stops
    return {"runs": n, "complete": runs.count("complete"), "runs_matching_pct": round(100 * runs.count("complete") / n, 1) if n else None,
            "judged": judged, "judged_matching_pct": round(100 * runs.count("complete") / judged, 1) if judged else None,
            "components": {c: {"floors": v[0], "matching": v[1], "pct": round(100 * v[1] / v[0], 1) if v[0] else None}
                           for c, v in sorted(comp.items())},
            "stops": dict(stops.most_common(15)), "commit": git_commit(emu)}


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


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("--version", default="v1")
    ap.add_argument("--subset", choices=("all", "pilot"), default="all")
    ap.add_argument("--emu", type=Path, default=ROOT, help="emulator checkout with EmuDump built")
    ap.add_argument("--procs", type=int, default=4)
    ap.add_argument("--label", default=None, help="a name for this score (default: the commit)")
    ap.add_argument("--codex", action="store_true", help="also walk the codex runs (whole runs, run-level offers)")
    ap.add_argument("--codex-emu", type=Path, default=None, help="checkout with the codex replay (default --emu)")
    ap.add_argument("--codex-limit", type=int, default=0)
    a = ap.parse_args()
    vdir = REFSET / a.version
    latest = json.loads((vdir / f"refs-{a.subset}.json").read_text(encoding="utf-8"))
    rdir = Path(latest["record"])
    manifest = json.loads((vdir / "manifest.json").read_text(encoding="utf-8"))
    entries = {e["id"]: e for e in manifest["specs"]}
    emu = a.emu.resolve()
    dll = emu / "tools" / "emudump" / "bin" / "Release" / "net9.0" / "EmuDump.dll"
    if not dll.exists():
        print(f"no EmuDump build in {emu}: dotnet build -c Release in tools/emudump first", file=sys.stderr)
        return 2
    emu_info = git_commit(emu) | {"emudump_mtime": time.strftime("%Y-%m-%d %H:%M", time.localtime(dll.stat().st_mtime))}
    nm = fc.names(json.loads((ROOT / "data" / "id_map.json").read_text(encoding="utf-8")))
    work = vdir / "scores" / "work"
    work.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    verdicts = []
    for part in latest["parts"]:
        p = rdir / f"{part}.refs.json"
        if not p.exists():
            continue
        games = json.loads(p.read_text(encoding="utf-8"))["results"]
        _, emus = emudump(dll, "fight", games, work, a.procs) or [None, []]
        rmeta, reas = emudump(dll, "reanchor", games, work, a.procs) or [{}, []]
        keys = rc.keys_of(rmeta)
        for g, e, r in zip(games, emus, reas):
            verdicts.append(judge(part, g, e, r, nm, keys))
    for v in verdicts:
        st = entries.get(v["id"], {}).get("strata", {})
        v["act"], v["room"] = st.get("act"), st.get("room")
        v["class"] = entries.get(v["id"], {}).get("class")
    by = collections.defaultdict(list)
    for v in verdicts:
        by[v["part"]].append(v)

    def block(vs):
        clean = [v for v in vs if not v.get("mirror_gap") and not v.get("setup_deck") and not v.get("setup_first")]
        return {"fights": len(vs), "identical": sum(1 for v in vs if v.get("identical")),
                "setup_matched": len(clean), "setup_matched_identical": sum(1 for v in clean if v.get("identical")),
                "transitions": sum(v.get("transitions", 0) for v in vs), "verified": sum(v.get("verified", 0) for v in vs),
                "mirror_gaps": sum(1 for v in vs if v.get("mirror_gap"))}
    real = by["realistic"]
    table = {}
    for act in (1, 2, 3):
        for room in ("monster", "elite", "boss"):
            vs = [v for v in real if v["act"] == act and v["room"] == room]
            if vs:
                table[f"act{act} {room}"] = block(vs)
    for room in ("monster", "elite", "boss"):
        vs = [v for v in real if v["room"] == room]
        if vs:
            table[f"all {room}"] = block(vs)
    classes = collections.defaultdict(list)
    for v in by["regression"]:
        classes[v["class"]].append(v)
    reg_pass = sorted(c for c, vs in classes.items() if all(v.get("identical") for v in vs))
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
    codex = codex_score((a.codex_emu or emu).resolve(), a.procs, a.codex_limit, work) if a.codex else {"skipped": "not run (--codex)"}
    runs = runs_score(vdir, emu, work)
    head = {p: block(by[p]) for p in PARTS if by[p]}
    commit = emu_info["commit"][:12] + ("-dirty" if emu_info["dirty"] else "")
    name = a.label or commit
    score = {"set": f"{manifest['set']} {manifest['version']}", "subset": a.subset, "emulator": emu_info, "label": name,
             "comparer": git_commit(ROOT), "record": latest["stamp"], "scored": time.strftime("%Y-%m-%d %H:%M"),
             "minutes": round((time.time() - t0) / 60, 1), "headline": head, "realistic_table": table,
             "regression": {"classes": len(classes), "passing": len(reg_pass), "failing": sorted(set(classes) - set(reg_pass))},
             "coverage": cov, "codex": codex, "runs": runs, "open_bug_classes": open_bugs, "verdicts": verdicts}
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
        f.write(json.dumps({"label": name, "subset": a.subset, "commit": emu_info["commit"], "scored": score["scored"],
                            "headline": head, "regression": score["regression"]["passing"], "line": line,
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
        x = f"{p} {b['identical']}/{b['fights']} identical ({pct(b['identical'], b['fights'])})"
        if steps and b.get("setup_matched", b["fights"]) < b["fights"]:
            x += f" [{pct(b['setup_matched_identical'], b['setup_matched'])} of the {b['setup_matched']} whose setup the emulator matched]"
        return x + (f", steps verified {pct(b['verified'], b['transitions'])}" if steps else "")
    c = s["coverage"]["counts"]
    nc = sum(v["not_clean"] + v["not_reached"] for v in c.values())
    cx = s["codex"]
    codex = (f"codex runs {cx['complete']}/{cx['runs']} matching to the end ({cx['runs_matching_pct']}%; "
             f"{cx['judged_matching_pct']}% of {cx['judged']} not stopped by a walker/fight/profile limit)" if cx.get("runs") else "codex runs n/a")
    rn = s.get("runs") or {}
    runs = (f"whole runs {rn['runs_matching']}/{rn['runs']} matching, {rn.get('fights_identical')}/{rn.get('fights')} of their fights identical"
            if rn.get("runs") else "whole runs n/a")
    return (f"{s['set']} [{s['subset']}] @ {s['label']}: {part('realistic', True)}; {part('coverage')}, {nc} items not yet clean; "
            f"regression {s['regression']['passing']}/{s['regression']['classes']} classes pass; {codex}; {runs}; "
            f"{len(s['open_bug_classes'])} open bug classes")


def delta(old: dict, new: dict) -> str:
    out = []
    for p in PARTS:
        if p in old and p in new:
            a, b = old[p], new[p]
            out.append(f"{p} identical {pct(a['identical'], a['fights'])} -> {pct(b['identical'], b['fights'])}, "
                       f"verified {pct(a['verified'], a['transitions'])} -> {pct(b['verified'], b['transitions'])}")
    return "; ".join(out)


def render_table(s: dict) -> str:
    rows = [f"  {'realistic':<14} {'fights':>6} {'identical':>10} {'steps':>7} {'verified':>9}"]
    for k, b in s["realistic_table"].items():
        rows.append(f"  {k:<14} {b['fights']:>6} {pct(b['identical'], b['fights']):>10} {b['transitions']:>7} {pct(b['verified'], b['transitions']):>9}")
    for p, b in s["headline"].items():
        rows.append(f"  {('= ' + p):<14} {b['fights']:>6} {pct(b['identical'], b['fights']):>10} {b['transitions']:>7} "
                    f"{pct(b['verified'], b['transitions']):>9}" + (f"  ({b['mirror_gaps']} mirror gaps)" if b["mirror_gaps"] else "")
                    + (f"  setup matched {b['setup_matched']}: {pct(b['setup_matched_identical'], b['setup_matched'])} identical"
                       if b.get("setup_matched", b["fights"]) < b["fights"] else ""))
    c = s["coverage"]["counts"]
    rows.append("  coverage items (whole set): " + "; ".join(
        f"{k} {v['clean']}/{v['items']} clean ({v['not_reached']} not reached, {v['not_clean']} diverging)" for k, v in c.items()))
    rows.append("  open bug classes (first divergence, fights): " + "; ".join(
        f"{b['cause']} {b['fights']}" for b in s["open_bug_classes"][:12]))
    if s["regression"]["failing"]:
        rows.append(f"  regression failing: {', '.join(s['regression']['failing'][:12])}")
    cx = s["codex"]
    if cx.get("components"):
        rows.append("  codex floors matching: " + "; ".join(f"{k} {v['pct']}%" for k, v in cx["components"].items()))
    return "\n".join(rows)


if __name__ == "__main__":
    sys.exit(main())
