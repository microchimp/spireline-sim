"""Turn a game-vs-emulator fight comparison into one compact markdown pack per likely bug.

    python -m tools.fidelity.context_pack <referee out.json> <emudump fight.json> [--out DIR] [--max-groups 20]

Each fight's first diverging step is found with fight_compare's views and diff. Fights are grouped
by cause signature: (differing fields, the card of the action that led to the divergence or
"end_turn"/"start", the enemies present). The differing fields are the top-level view fields, with
`player` and `rng` split one level (player.hp, rng.Shuffle) because those are different causes.
Each group gets a pack: one representative divergence, the entities involved, and where each
entity lives in the decompiled game and in the emulator (paths only, no code). Decompiled paths
are shown relative to the decompiled root, emulator paths relative to the repo root. When
referee/trace_divergent.py has traced a fight of the group (<out>/traces/<seed>.trace.txt), its trace
diff (the first event that differs inside the diverging step) is appended to the pack.
"""

from __future__ import annotations

import argparse
import collections
import json
import re
import sys
from pathlib import Path

from tools.fidelity.fight_compare import emu_view, first_difference, game_view, names, snake

ROOT = Path(__file__).resolve().parents[2]
DECOMPILED = Path(r"D:\spireline\referee\decompiled")
MODELS = "MegaCrit.Sts2.Core.Models."
EMU_FILES = ["Core/Effects/CardEffects.cs", "Core/Effects/PotionEffects.cs", "Core/Effects/RelicEffects.cs",
             "Core/EnemyAI.cs", "Core/BuffState.cs", "Core/CombatEngine.cs", "Core/CombatFactory.cs"]
EMU_SRC = ROOT / "src" / "Sts2Emulator"
PILES = ("hand", "draw", "discard", "exhaust")
MAX_LINES = 60
MAX_TRACE_LINES = 25


def norm(s: str) -> str:
    return re.sub(r"[^a-z0-9]", "", s.lower())


# ---------------------------------------------------------------- first divergence

def first_divergence(g: dict, e: dict, nm: dict):
    """(step index, action, diffs, game view, emulator view, previous actions) or None."""
    fd = first_difference(g, e, nm)
    if fd is None:
        return None
    i, d = fd
    gsteps, esteps = g["steps"], e.get("steps", [])
    gv = game_view(gsteps[i]["state"])
    ev = emu_view(esteps[i]["state"], nm) if i < len(esteps) else {"in_combat": False}
    prev = [gsteps[j]["action"] for j in range(max(0, i - 2), i)]
    return i, gsteps[i]["action"], d, gv, ev, prev


def top_fields(d, gv=None, ev=None) -> tuple:
    out = set()
    for path, _, _ in d:
        parts = path.strip(".").split(".")
        if parts[0] == "enemies" and gv is not None:  # split by what differs: hp, block, powers, presence
            out |= {"enemies." + k for _, _, fs in enemy_diffs(gv, ev) for k, _, _ in fs}
            continue
        out.add(".".join(parts[:2]) if parts[0] in ("player", "rng") and len(parts) > 1 else parts[0])
    return tuple(sorted(out))


def action_card(act) -> str:
    if act is None:
        return "start"
    return act.get("card") or act.get("kind", "?")


# ---------------------------------------------------------------- views to text

def fmt_card(c) -> str:
    return c[0] + ("+" if c[1] else "")


def fmt_powers(ps) -> str:
    return ", ".join(f"{n}:{a}" for n, a in ps) or "-"


def fmt_act(a) -> str:
    if a is None:
        return "start"
    if a.get("kind") == "play":
        t = a.get("target_slot", -1)
        return f"play {a['card']}" + (f" -> slot {t}" if t is not None and t >= 0 else "")
    return a.get("kind", "?")


def fmt_list(x, y, fmt) -> list[str]:
    """Game and emulator lists trimmed to the differing region (common prefix/suffix elided)."""
    p = 0
    while p < min(len(x), len(y)) and x[p] == y[p]:
        p += 1
    s = 0
    while s < min(len(x), len(y)) - p and x[-1 - s] == y[-1 - s]:
        s += 1
    def region(v):
        r = v[p:len(v) - s]
        txt = " ".join(fmt(c) for c in r[:12]) + (f" ...(+{len(r) - 12})" if len(r) > 12 else "")
        return f"[{txt}]"
    note = f" (same: first {p}, last {s})" if p or s else ""
    return [f"game {len(x)}: {region(x)}", f"emu  {len(y)}: {region(y)}{note}"]


def enemy_diffs(gv, ev) -> list[tuple]:
    """(index, name, [(field, game, emu)]) for enemies that differ, by position."""
    out = []
    ge, ee = gv.get("enemies", []), ev.get("enemies", [])
    for i in range(max(len(ge), len(ee))):
        a, b = (ge[i] if i < len(ge) else None), (ee[i] if i < len(ee) else None)
        if a is None or b is None:
            out.append((i, (a or b)["name"], [("present", "yes" if a else "no", "yes" if b else "no")]))
            continue
        fs = [(k, a.get(k), b.get(k)) for k in ("name", "hp", "block", "powers") if a.get(k) != b.get(k)]
        if fs:
            out.append((i, a["name"], fs))
    return out


def power_diffs(pg, pe) -> list[tuple]:
    a, b = dict(map(tuple, pg)), dict(map(tuple, pe))
    return [(n, a.get(n), b.get(n)) for n in sorted(set(a) | set(b)) if a.get(n) != b.get(n)]


def pile_only(gv, ev) -> list[tuple]:
    """(pile, card, game count, emu count) for cards whose count differs in a pile."""
    out = []
    for pile in PILES:
        a = collections.Counter(fmt_card(c) for c in gv.get(pile, []))
        b = collections.Counter(fmt_card(c) for c in ev.get(pile, []))
        out += [(pile, c, a[c], b[c]) for c in sorted(set(a) | set(b)) if a[c] != b[c]]
    return out


# ---------------------------------------------------------------- source locations

class Locator:
    def __init__(self, decompiled: Path, nm_pascal: dict):
        self.root = decompiled
        self.pascal = nm_pascal  # {kind: {norm(name): PascalName}}
        self.dirs = {}
        self.emu = {}

    def _dir(self, suffix: str) -> dict:
        if suffix not in self.dirs:
            d = self.root / (MODELS + suffix)
            self.dirs[suffix] = {norm(p.stem): p for p in d.glob("*.cs")} if d.is_dir() else {}
        return self.dirs[suffix]

    def game(self, kind: str, snk: str) -> list[str]:
        suffix = {"card": "Cards", "power": "Powers", "monster": "Monsters", "relic": "Relics"}[kind]
        n = norm(snk)
        idx = self._dir(suffix)
        hit = idx.get(n + "power") if kind == "power" else idx.get(n)
        if hit is None and kind == "power":
            hit = idx.get(n)
        if hit is not None:
            return [hit.relative_to(self.root).as_posix()]
        pas = self.pascal_name(kind, snk)  # fall back to grep -l "class <Pascal>"
        d = self.root / (MODELS + suffix)
        found = []
        if d.is_dir():
            pat = re.compile(r"\bclass\s+" + re.escape(pas + ("Power" if kind == "power" else "")) + r"\b")
            for p in sorted(d.glob("*.cs")):
                try:
                    if pat.search(p.read_text(encoding="utf-8", errors="replace")):
                        found.append(p.relative_to(self.root).as_posix())
                except OSError:
                    pass
        return found[:3]

    def pascal_name(self, kind: str, snk: str) -> str:
        table = self.pascal.get(kind, {})
        return table.get(norm(snk), "".join(w.capitalize() for w in snk.split("_")))

    def emulator(self, kind: str, snk: str) -> list[str]:
        pas = re.escape(self.pascal_name(kind, snk))
        prefix = {"card": r"IC\.", "monster": r"KE\.", "power": r"BuffId\."}[kind]
        pats = [re.compile(r"\b" + prefix + pas + r"\b")]
        if kind != "card":
            pats.append(re.compile(r"\b" + pas + r"\b"))
        hits: list[str] = []
        for pat in pats:
            for rel in EMU_FILES:
                for ln, line in self._lines(rel):
                    if pat.search(line):
                        hits.append(f"src/Sts2Emulator/{rel}:{ln}")
                        if len(hits) >= 5:
                            return hits
            if hits:
                break
        return hits

    def _lines(self, rel: str):
        if rel not in self.emu:
            p = EMU_SRC / rel
            try:
                self.emu[rel] = list(enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines(), 1))
            except OSError:
                self.emu[rel] = []
        return self.emu[rel]


def pascal_tables(id_map: dict) -> dict:
    def tab(section, strip=""):
        out = {}
        for k in id_map.get(section, {}):
            n = norm(k)
            out[n[: -len(strip)] if strip and n.endswith(strip) else n] = k[: -len(strip)] if strip and k.endswith(strip.capitalize()) else k
        return out
    return {"card": tab("cards"), "monster": tab("enemies"), "power": tab("powers", "power"), "relic": tab("relics")}


# ---------------------------------------------------------------- packs

def trace_section(seeds: list, traces: Path | None) -> list[str]:
    """The trace diff (referee/trace_divergent.py) of the first seed of the group that has one."""
    if traces is None or not traces.is_dir():
        return []
    for seed in seeds:
        f = traces / f"{seed}.trace.txt"
        if f.is_file():
            body = f.read_text(encoding="utf-8").splitlines()
            cut = body[:MAX_TRACE_LINES] + (["...(truncated)"] if len(body) > MAX_TRACE_LINES else [])
            return ["", f"## Trace diff (seed {seed}; referee/trace_divergent.py)", "```", *cut, "```"]
    return []


def minimal_section(items, minimal: dict, flags: dict) -> list:
    """Setup fights (referee/scenarios.py): the plausibility flags and the shrunk setups of the group's fights."""
    L = []
    flagged = {it["seed"]: flags.get(it["seed"]) for it in items if flags.get(it["seed"])}
    if flagged:
        L += ["", "## Plausibility", f"- {len(flagged)} of {len(items)} fights need a setup no real run has: "
              + "; ".join(f"{s} {','.join(f)}" for s, f in list(flagged.items())[:6])]
    shrunk = [minimal[it["seed"]] for it in items if it["seed"] in minimal]
    if shrunk:
        L += ["", "## Minimal setup (referee/shrink.py; spec in minimal/<seed>.json)"]
        for m in shrunk[:4]:
            L.append(f"- {m['seed']}: {len(m['items'])} of {m['of']} items still diverge at step {m['step']} ({m['field']}), "
                     f"{m['status']} after {m['fights']} game fights: "
                     + (", ".join(f"{it.get('cmd')} {it.get('id')}" + (f" {it.get('amount')}@{it.get('target')}" if it.get("cmd") == "power" else "")
                                  + ("+" if it.get("upgraded") else "") for it in m["items"]) or "none: the fight item alone"))
    return L


def build_pack(sig, items, loc: Locator, nm, traces: Path | None = None, minimal: dict | None = None,
               flags: dict | None = None) -> str:
    fields, card, enemies = sig
    rep = min(items, key=lambda it: it["step"])
    gv, ev, d = rep["gv"], rep["ev"], rep["diffs"]
    seeds = [it["seed"] for it in items]
    L = [f"# {', '.join(fields)} after {card} vs {', '.join(enemies)}", "",
         f"- signature: fields=[{', '.join(fields)}] action={card} enemies=[{', '.join(enemies)}]",
         f"- count: {len(items)} fights; seeds: {', '.join(seeds[:8])}" + (f" (+{len(seeds) - 8} more)" if len(seeds) > 8 else ""),
         f"- representative: seed {rep['seed']}, first difference at step {rep['step']}", "",
         "## Representative", f"- action: {fmt_act(rep['action'])}",
         f"- preceding: {'; '.join(fmt_act(a) for a in rep['prev']) or '(none)'}", "- differing fields (game vs emulator):"]
    ens = []  # entities: (kind, snake name, why)
    if rep["action"] and rep["action"].get("card"):
        ens.append(("card", rep["action"]["card"], "action"))
    for path, x, y in d:
        top = path.strip(".").split(".")[0]
        if top in PILES:
            L.append(f"  - {top}:")
            L += [f"      {t}" for t in fmt_list(x, y, fmt_card)]
        elif top == "enemies":
            continue
        elif top == "player" and path.endswith(".powers"):
            L.append(f"  - player.powers: game {fmt_powers(x)}  emu {fmt_powers(y)}")
            ens += [("power", n, "player") for n, _, _ in power_diffs(x, y)]
        else:
            L.append(f"  - {path.strip('.')}: game {x}  emu {y}")
    for i, name, fs in enemy_diffs(gv, ev):
        L.append(f"  - enemy[{i}] {name}:")
        ens.append(("monster", name, "enemy differs"))
        for k, x, y in fs:
            if k == "powers":
                L.append(f"      powers: game {fmt_powers(x)}  emu {fmt_powers(y)}")
                ens += [("power", n, f"on {name}") for n, _, _ in power_diffs(x, y)]
            else:
                L.append(f"      {k}: game {x}  emu {y}")
    po = pile_only(gv, ev)
    if po:
        L.append("- cards whose count differs in a pile (pile card game/emu):")
        L.append("  " + "; ".join(f"{p} {c} {a}/{b}" for p, c, a, b in po[:10]) + (f" ...(+{len(po) - 10})" if len(po) > 10 else ""))
        for _, c, _, _ in po[:6]:
            ens.append(("card", re.sub(r"\+$", "", c), "only on one side of a pile"))
    seen, uniq = set(), []
    for k, n, why in ens:
        if (k, n) not in seen:
            seen.add((k, n))
            uniq.append((k, n, why))
    L += ["", "## Entities and sources (paths only; decompiled paths are relative to the decompiled root)"]
    for k, n, why in uniq[:12]:
        L.append(f"- {k} {n} ({why})")
        L.append(f"  - game: {', '.join(loc.game(k, n)) or 'not found'}")
        L.append(f"  - emulator: {', '.join(loc.emulator(k, n)) or 'no match'}")
    if len(uniq) > 12:
        L.append(f"- ...(+{len(uniq) - 12} more entities)")
    extra = minimal_section(items, minimal or {}, flags or {})  # kept whole: it is the triage answer
    if len(L) + len(extra) > MAX_LINES:
        L = L[: max(10, MAX_LINES - 1 - len(extra))] + ["...(truncated)"]
    L += extra
    L += trace_section(seeds, traces)
    return "\n".join(L) + "\n"


def slug(sig) -> str:
    return re.sub(r"[^A-Za-z0-9]+", "-", f"{'+'.join(sig[0])}-{sig[1]}").strip("-").lower()[:60]


def run(game_path, emu_path, out: Path, max_groups=20, id_map_path=None, decompiled=DECOMPILED,
        minimal: dict | None = None, flags: dict | None = None) -> str:
    """minimal and flags (by seed) come from setup fights; groups with a plausible fight are listed first."""
    id_map = json.loads(Path(id_map_path or ROOT / "data" / "id_map.json").read_text(encoding="utf-8"))
    nm = names(id_map)
    games = json.loads(Path(game_path).read_text(encoding="utf-8"))["results"]
    emus = json.loads(Path(emu_path).read_text(encoding="utf-8"))["results"]
    groups = collections.defaultdict(list)
    same = errors = 0
    for g, e in zip(games, emus):
        if g.get("error") or e.get("error"):
            errors += 1
        r = first_divergence(g, e, nm)
        if r is None:
            same += 1
            continue
        i, act, d, gv, ev, prev = r
        enemies = tuple(sorted({x["name"] for x in gv.get("enemies", [])}))
        groups[(top_fields(d, gv, ev), action_card(act), enemies)].append(
            {"seed": g.get("seed"), "step": i, "action": act, "diffs": d, "gv": gv, "ev": ev, "prev": prev})
    out.mkdir(parents=True, exist_ok=True)
    for old in out.glob("g[0-9][0-9]_*.md"):
        old.unlink()
    loc = Locator(Path(decompiled), pascal_tables(id_map))
    flags = flags or {}
    ordered = sorted(groups.items(), key=lambda kv: (all(flags.get(it["seed"]) for it in kv[1]), -len(kv[1]), kv[0]))
    total = sum(len(v) for _, v in ordered)
    idx = [f"# Fidelity context packs", "",
           f"{len(games)} fights: {same} identical, {total} diverging in {len(ordered)} groups, {errors} with errors.", ""]
    for n, (sig, items) in enumerate(ordered[:max_groups], 1):
        fname = f"g{n:02d}_{slug(sig)}.md"
        (out / fname).write_text(build_pack(sig, items, loc, nm, out / "traces", minimal, flags), encoding="utf-8")
        idx.append(f"{n}. [{len(items)}] {', '.join(sig[0])} after {sig[1]} vs {', '.join(sig[2])} "
                   f"- {fname} (e.g. {items[0]['seed']})")
    if len(ordered) > max_groups:
        rest = sum(len(v) for _, v in ordered[max_groups:])
        idx.append(f"... {len(ordered) - max_groups} more groups ({rest} fights) not written; raise --max-groups")
    text = "\n".join(idx) + "\n"
    (out / "index.md").write_text(text, encoding="utf-8")
    return text


def main() -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    ap.add_argument("referee")
    ap.add_argument("emulator")
    ap.add_argument("--out", default=None, help="output directory (default: <emulator json without .emu.json>.packs next to it)")
    ap.add_argument("--max-groups", type=int, default=20)
    ap.add_argument("--id-map", default=None)
    ap.add_argument("--decompiled", default=str(DECOMPILED))
    a = ap.parse_args()
    out = Path(a.out) if a.out else Path(a.emulator).with_suffix("").with_suffix("").with_name(Path(a.emulator).name.split(".")[0] + ".packs")
    sys.stdout.write(run(a.referee, a.emulator, out, a.max_groups, a.id_map, a.decompiled))
    return 0


if __name__ == "__main__":
    sys.exit(main())
