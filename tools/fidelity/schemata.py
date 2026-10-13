"""Mutant schemata: every sampled mutant compiled into one engine build, each behind ``MutCov.Is(id)``.

    python -m tools.fidelity.mutate schemata --dir <run dir> --reanchor <reanchor.json> [--mutants 150] [--shards 6]

One build instead of one per mutant, and one long-lived ``EmuDump muthost`` process per shard of fights
instead of one per mutant and mode: per fight the run is set up once (RunEngine.Reset, most of a replay's
cost, holds no mutant) and every mutant whose block the fight ran is replayed from it. The kill test is the
plain ``run`` command's (mutate.Judge). Fights go in one shared random order, split round-robin across the
shards; a mutant is no longer run on fights after its first kill (the shards read a kills file).

The guarded form of each operator: an expression mutant becomes ``(Is(id) ? <mutant> : <original>)``
over its operands (found on the masked line by precedence; a mutant whose operands cannot be found on one
line is not built), a deleted call ``{ if (!Is(id)) <call> }``, a swap ``if (Is(id)) { B A } else { A`` /
``B }``. Line numbers never move. A guarded mutant that does not compile is dropped as not built.
"""

from __future__ import annotations

import collections
import json
import math
import random
import re
import threading
import time
from pathlib import Path

from . import mutate as mu

IS = "global::Sts2Emulator.MutCov.Is({})"
MUTCOV = """// Written by tools/fidelity/mutate.py into its scratch copy only: probes, the active mutant, the setup memo.
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;

namespace Sts2Emulator;

public static class MutCov
{{
    static readonly bool[] Seen = new bool[{n}];
    static readonly Dictionary<(string, int), RunState> Memo = new();
    public static int Active = -1;
    public static bool NoMemo;

    public static void Hit(int i) => Seen[i] = true;

    public static bool Is(int id) => Active == id;

    public static int[] Take()
    {{
        var got = new List<int>();
        for (int i = 0; i < Seen.Length; i++)
        {{
            if (Seen[i]) {{ got.Add(i); Seen[i] = false; }}
        }}

        return [.. got];
    }}

    public static void Forget() => Memo.Clear();

    public static RunState? Recall(string seed, int asc) =>
        !NoMemo && Memo.TryGetValue((seed, asc), out var s) ? s.Clone() : null;

    public static void Remember(string seed, int asc, RunState s)
    {{
        if (!NoMemo && Active == -1 && !Memo.ContainsKey((seed, asc))) Memo[(seed, asc)] = s.Clone();
    }}
}}
"""
# RunEngine.Reset remembers its result per seed and hands back a copy (scratch copy only).
MEMO_PATCHES = [
    ("    public RunState State { get; private init; } = new();", "    public RunState State { get; private set; } = new();"),
    ("    public void Reset(string seed, int ascension = Ascension.DefaultLevel)\n    {",
     "    public void Reset(string seed, int ascension = Ascension.DefaultLevel)\n    { if (global::Sts2Emulator.MutCov.Recall(seed,"
     " ascension) is { } __memo) { State = __memo; return; }"),
    ("        State.EventSequenceIndex = 1;\n    }\n",
     "        State.EventSequenceIndex = 1; global::Sts2Emulator.MutCov.Remember(seed, ascension, State);\n    }\n"),
]
KEYWORDS = ("return", "when", "case", "throw", "in", "else", "yield", "await")
PATTERN_WORDS = re.compile(r"\b(is|and|or|not)\b")
# What ends an operand of a relational (or, for == and !=, an equality) operator: anything of lower precedence.
STOP_REL = ("==", "!=", "&&", "||", "??", "=>")
STOP_EQ = ("&&", "||", "??", "=>")


def _starts_expression(prev: str) -> bool:
    """Whether a line can start a fresh expression after this (masked) line."""
    p = prev.rstrip()
    return not p or p.endswith(("(", ",", "&&", "||", "?", ":", "=", "{", ";", "}", "=>", "["))


def operand_left(m: str, s: int, eq: bool, prev: str) -> int | None:
    """Start column of the operand that ends just before column s, or None if not on this line."""
    stops = STOP_EQ if eq else STOP_REL
    depth, k = 0, s - 1
    while k >= 0:
        ch = m[k]
        two = m[k - 1:k + 1] if k >= 1 else ""
        if ch in ")]":
            depth += 1
        elif ch in "([":
            if depth == 0:
                break
            depth -= 1
        elif depth == 0:
            if two in stops or (two in ("<=", ">=") and not eq):
                break
            if ch in ",;{}" or (ch == "?" and m[k + 1:k + 2] not in (".", "[")) or ch == ":":
                break
            if ch in "&|^" and two not in ("&&", "||"):
                break
            if ch == "=" and m[k - 1:k] not in ("=", "!", "<", ">") and m[k + 1:k + 2] not in ("=", ">"):
                break
            if ch in "<>" and m[k + 1:k + 2] != "=" and m[k - 1:k] not in ("=", "-") and not eq:
                return None  # another relational operator (or a generic): give up
        k -= 1
    start = k + 1
    text = m[start:s]
    lead = re.match(r"\s*(" + "|".join(KEYWORDS) + r")\b\s*", text)
    if lead:
        start += lead.end()
    if k < 0 and not lead and not _starts_expression(prev):
        return None
    body = m[start:s].strip()
    if not body or PATTERN_WORDS.search(body):
        return None
    return start + (len(m[start:s]) - len(m[start:s].lstrip()))


def operand_right(m: str, e: int, eq: bool, nxt: str) -> int | None:
    """End column of the operand that starts just after column e, or None if not on this line."""
    stops = STOP_EQ if eq else STOP_REL
    depth, k = 0, e
    while k < len(m):
        ch = m[k]
        two = m[k:k + 2]
        if ch in "([":
            depth += 1
        elif ch in ")]":
            if depth == 0:
                break
            depth -= 1
        elif depth == 0:
            if two in stops or (two in ("<=", ">=") and not eq):
                break
            if ch in ",;{}:" or (ch == "?" and m[k + 1:k + 2] not in (".", "[")):
                break
            if ch in "&|^" and two not in ("&&", "||"):
                break
            if ch == "=" and m[k + 1:k + 2] not in ("=", ">") and m[k - 1:k] not in ("=", "!", "<", ">"):
                break
            if ch in "<>" and not eq and m[k + 1:k + 2] != "=" and m[k - 1:k] != "=":
                return None
        k += 1
    if k >= len(m.rstrip()) and not nxt.strip().startswith(("&&", "||", ")", "?", ":", "{", ";", ",")):
        return None  # the operand goes on past the end of the line
    end = k
    body = m[e:end].strip()
    if not body or PATTERN_WORDS.search(body):
        return None
    return e + len(m[e:end].rstrip())


def receiver_left(m: str, dot: int) -> int | None:
    """Start of the member-access chain ending at the `.` in column dot (`state.ActiveCombat!.ShuffleRng`)."""
    depth, k = 0, dot - 1
    while k >= 0:
        ch = m[k]
        if ch in ")]":
            depth += 1
        elif ch in "([":
            if depth == 0:
                break
            depth -= 1
        elif depth == 0 and not (ch.isalnum() or ch in "_.!?"):
            break
        k -= 1
    start = k + 1
    return start if start < dot else None


def guard(text: str, muts: list) -> tuple:
    """The text with every mutant behind Is(id) (muts: [(id, Mutant)], at most one per line); (text, skipped)."""
    lines = text.split("\n")
    mlines = mu.mask(text).split("\n")
    skipped = {}
    for gid, m in muts:
        i = m.line - 1
        line, ml = lines[i], mlines[i]
        if line != m.old.split("\n")[0]:
            skipped[gid] = "source moved"
            continue
        x = m.extra
        cond = IS.format(gid)
        if m.op in ("cmp-boundary", "cmp-negate"):
            eq = x["from"] in ("==", "!=")
            ls = operand_left(ml, x["col"], eq, mlines[i - 1] if i else "")
            re_ = operand_right(ml, x["end"], eq, mlines[i + 1] if i + 1 < len(mlines) else "")
            if ls is None or re_ is None:
                skipped[gid] = "operands not on one line"
                continue
            left, right = line[ls:x["col"]], line[x["end"]:re_]
            lines[i] = (line[:ls] + f"({cond} ? ({left}{x['rep']}{right}) : ({left}{x['from']}{right}))" + line[re_:])
        elif m.op in ("literal+1", "literal-1", "drop-clamp", "drop-rounding"):
            orig = line[x["col"]:x["end"]]
            lines[i] = line[:x["col"]] + f"({cond} ? {x['rep']} : {orig})" + line[x["end"]:]
        elif m.op == "rng-stream":
            rs = receiver_left(ml, x["col"] - 1)
            if rs is None:
                skipped[gid] = "no receiver"
                continue
            rec = line[rs:x["col"]]
            lines[i] = line[:rs] + f"({cond} ? {rec}{x['rep']} : {rec}{x['from']})" + line[x["end"]:]
        elif m.op == "delete-call":
            body = line.rstrip("\r")
            ind = body[: len(body) - len(body.lstrip())]
            lines[i] = f"{ind}{{ if (!{cond}) {body.strip()} }}" + ("\r" if line.endswith("\r") else "")
        elif m.op == "swap-calls":
            a, b = line.rstrip("\r"), lines[i + 1].rstrip("\r")
            if b != m.old.split("\n")[1].rstrip("\r"):
                skipped[gid] = "source moved"
                continue
            ind = a[: len(a) - len(a.lstrip())]
            cr = "\r" if line.endswith("\r") else ""
            lines[i] = f"{ind}if ({cond}) {{ {b.strip()} {a.strip()} }} else {{ {a.strip()}" + cr
            lines[i + 1] = f"{ind}{b.strip()} }}" + cr
        else:
            skipped[gid] = f"no guarded form for {m.op}"
    return "\n".join(lines), skipped


def pick(cands: list, k: int, rng: random.Random) -> list:
    """mutate.sample, keeping one mutant per line (a swap holds two)."""
    taken, out = set(), []
    for m in mu.sample(cands, len(cands), rng):
        span = {(m.file, m.line), (m.file, m.line + 1)} if m.op == "swap-calls" else {(m.file, m.line)}
        if span & taken:
            continue
        taken |= span
        out.append(m)
        if len(out) == k:
            break
    return out


def build_schemata(copy: Path, muts: list, log=print) -> tuple:
    """Write and build the guarded, probed engine; ({gid: unit}, {gid: why not built})."""
    mu.make_copy(mu.ROOT, copy)
    by_file = collections.defaultdict(list)
    for gid, m in muts:
        by_file[m.file].append((gid, m))
    dropped, skip_probe = {}, collections.defaultdict(set)
    eng = mu.ROOT / mu.ENGINE / "Core/Run/RunEngine.cs"
    text = eng.read_bytes().decode("utf-8")
    crlf = "\r\n" in text
    text = text.replace("\r\n", "\n")
    for a, b in MEMO_PATCHES:
        assert text.count(a) == 1, f"RunEngine.cs moved: {a[:60]}"
        text = text.replace(a, b)
    (copy / mu.ENGINE / "Core/Run/RunEngine.cs").write_bytes((text.replace("\n", "\r\n") if crlf else text).encode("utf-8"))
    for attempt in range(25):
        nid, units, lines_of = 0, {}, {}
        for rel in mu.TARGETS:
            clean = (mu.ROOT / mu.ENGINE / rel).read_bytes().decode("utf-8")
            live = [(g, m) for g, m in by_file[rel] if g not in dropped]
            text, skipped = guard(clean, live)
            dropped.update(skipped)
            pr = mu.instrument(text, nid, frozenset(skip_probe[rel]))
            (copy / mu.ENGINE / rel).write_bytes(pr.text.encode("utf-8"))
            for g, m in live:
                if g not in dropped:
                    units[g] = pr.unit[m.line - 1]
                    lines_of[rel, m.line] = g
                    if m.op == "swap-calls":
                        lines_of[rel, m.line + 1] = g
            nid += len(pr.probes)
            skip_probe[rel + "#probes"] = set(pr.probes.values())
        (copy / mu.ENGINE / "Core/MutCov.cs").write_text(MUTCOV.format(n=max(1, nid)), encoding="utf-8")
        ok, out = mu.build(copy, restore=attempt == 0)
        if ok:
            return units, dropped
        bad = set(re.findall(r"[\\/]src[\\/]Sts2Emulator[\\/](Core[\\/][\w\\/]+\.cs)\((\d+),\d+\): error (CS\d+)", out))
        if not bad:
            log(out[-3000:])
            raise RuntimeError("schemata build failed without a source error")
        n_mut = n_probe = 0
        for f, ln, code in bad:
            rel, ln = f.replace("\\", "/"), int(ln)
            g = lines_of.get((rel, ln))
            if g is not None:
                if g not in dropped:
                    dropped[g] = f"guarded form does not compile ({code})"
                    n_mut += 1
                continue
            # An error on a line without a mutant (a name it hid, a brace it broke) is the nearest one above's.
            near = [(ml, mg) for (mf, ml), mg in lines_of.items() if mf == rel and ml <= ln and mg not in dropped]
            if near:
                g = max(near)[1]
                dropped[g] = f"guarded form breaks line {ln} ({code})"
                n_mut += 1
                continue
            above = [p for p in skip_probe[rel + "#probes"] if p <= ln]
            if above and max(above) not in skip_probe[rel]:
                skip_probe[rel].add(max(above))
                n_probe += 1
        log(f"schemata build {attempt}: dropped {n_mut} mutants, {n_probe} probes", flush=True)
        if not n_mut and not n_probe:
            log(out[-3000:])
            raise RuntimeError("schemata build keeps failing")
    raise RuntimeError("schemata build did not converge")


def cmd_schemata(a) -> int:
    from .fight_compare import names  # noqa: PLC0415
    from .reanchor_compare import keys_of  # noqa: PLC0415

    d = Path(a.dir)
    prep = json.loads((d / "prepared.json").read_text(encoding="utf-8"))
    fights = prep["fights"]
    weight = {c["id"]: c["fights"] for c in prep["candidates"]}
    unit0 = {c["id"]: c["unit"] for c in prep["candidates"]}
    cands = []
    for rel in mu.TARGETS:  # again from the source, for the columns; same ids as prepare's
        for m in mu.sites(rel, (mu.ROOT / mu.ENGINE / rel).read_bytes().decode("utf-8")):
            m.fights, m.unit = weight.get(m.id, 0), unit0.get(m.id)
            cands.append(m)
    sd = d / f"schemata{a.tag}"
    sd.mkdir(exist_ok=True)
    chosen = pick(cands, a.mutants, random.Random(a.seed))
    muts = list(enumerate(chosen))
    t0 = time.time()
    units, dropped = build_schemata(sd / "src", muts)
    print(f"built {len(units)} of {len(muts)} mutants in {time.time() - t0:.0f} s", flush=True)
    nm = names(json.loads((mu.ROOT / "data/id_map.json").read_text(encoding="utf-8")))
    games = mu._load_games(prep["out_json"])
    keys = keys_of(prep["keys"])
    reanchored = json.loads(Path(a.reanchor).read_text(encoding="utf-8"))["results"]
    judge = mu.Judge(games, nm, keys, mu.base_verdicts(games, reanchored, nm, keys, fights))
    del reanchored
    perm = list(fights)
    random.Random(f"{a.seed}:order").shuffle(perm)
    pos = {i: p + 1 for p, i in enumerate(perm)}
    if a.max_fights:
        perm = perm[:a.max_fights]  # a trial: the first n of the order only
    live = {g: u for g, u in units.items() if u is not None}
    kills_path = sd / "kills.txt"
    kills_path.write_text("", encoding="utf-8")
    best = {}                   # gid -> first kill position
    first = collections.defaultdict(list)
    ran = collections.Counter()
    silent = collections.Counter()
    events = []
    lock = threading.Lock()
    deadline = time.time() + a.minutes * 60

    def kill(g, i, mode, why):
        p = pos[i]
        with lock:
            if g in best and best[g] < p:
                return
            if g not in best or p < best[g]:
                best[g] = p
                first[g] = []
                for _ in range(100):  # a host may be reading it
                    try:
                        with kills_path.open("a", encoding="utf-8") as f:
                            f.write(f"{g} {p}\n")
                        break
                    except OSError:
                        time.sleep(0.05)
            first[g].append({"i": i, "pos": p, "mode": mode, "why": why})

    def shard(part: list):
        todo = list(part)
        retries = collections.Counter()
        while todo and time.time() < deadline:
            args = ["dotnet", mu.emudump(sd / "src"), "muthost", prep["out_json"],
                    "--fights", ",".join(f"{i}:{pos[i]}" for i in todo),
                    "--mutants", ",".join(f"{g}:{u}" for g, u in live.items()),
                    "--baseline", str(d / "baseline"), "--kills", str(kills_path)]
            p = mu.popen(args)
            state = {"t": time.time(), "cur": None, "fight": None, "done": set()}

            def drain():
                for raw in p.stdout:
                    o = json.loads(raw)
                    with lock:
                        state["t"] = time.time()
                    if "base" in o:
                        state["fight"], state["cur"] = o["base"], None
                        if not o["ok"]:
                            events.append({"contaminated_at": o["base"]})
                    elif "b" in o:
                        state["cur"] = tuple(o["b"])
                    elif "m" in o:
                        i, g = o["i"], o["m"]
                        why = None
                        if o.get("fight") is not None:
                            v = judge.fight(i, o["fight"])
                            why = ("fight", v) if v else None
                        if not why and o.get("reanchor") is not None:
                            v = judge.reanchor(i, o["reanchor"])
                            why = ("reanchor", v) if v else None
                        if why:
                            kill(g, i, *why)
                        else:
                            with lock:
                                silent[g] += 1
                    elif "done" in o:
                        with lock:
                            for g in o["ran"]:
                                ran[g] += 1
                        state["done"].add(o["done"])
                        state["cur"] = None

            t = threading.Thread(target=drain, daemon=True)
            t.start()
            hung = False
            while t.is_alive():
                time.sleep(0.5)
                with lock:
                    quiet = time.time() - state["t"]
                if quiet > a.stall or time.time() > deadline:
                    hung = quiet > a.stall
                    p.kill()
                    break
            t.join()
            rc = p.wait()
            todo = [i for i in todo if i not in state["done"]]
            if not todo:
                break
            cur = state["cur"]
            if cur and (hung or rc not in (0, 3)):
                kill(cur[1], cur[0], "host", "timeout" if hung else "crash")
            elif rc == 3:
                retries[todo[0]] += 1
                if retries[todo[0]] > 1:
                    events.append({"skipped_fight": todo[0]})
                    todo = todo[1:]
            elif time.time() > deadline:
                break
            else:
                retries[todo[0]] += 1
                if retries[todo[0]] > 2:
                    events.append({"skipped_fight": todo[0], "rc": rc})
                    todo = todo[1:]

    parts = [perm[s::a.shards] for s in range(a.shards)]
    ts = [threading.Thread(target=shard, args=(pt,)) for pt in parts]
    for t in ts:
        t.start()
    last = time.time()
    while any(t.is_alive() for t in ts):
        time.sleep(2)
        if time.time() - last > 60:
            last = time.time()
            with lock:
                print(f"{time.time() - t0:.0f} s: {len(best)} of {len(live)} killed", flush=True)
    for t in ts:
        t.join()
    covering = collections.Counter(u for us in prep["fight_cov"].values() for u in us)
    with (sd / "results.jsonl").open("w", encoding="utf-8") as f:
        for g, m in muts:
            rec = {**{k: v for k, v in vars(m).items()}, "gid": g}
            if g in dropped:
                rec["status"], rec["why"] = "build_failed", dropped[g]
            elif units.get(g) is None:
                rec["status"] = "no_probe"
            else:
                rec.update(status="killed" if g in best else "survived", first_kill_pos=best.get(g),
                           first_kill=first.get(g, []), fights_run=ran[g], silent=silent[g],
                           fights_selected=covering.get(m.unit, 0) if m.unit is not None else None)
            f.write(json.dumps(rec) + "\n")
    (sd / "events.json").write_text(json.dumps(events), encoding="utf-8")
    print(f"done in {time.time() - t0:.0f} s; {len(best)} killed of {len(live)} built; events {len(events)}", flush=True)
    return 0


def kill_curve_shared(recs: list, ns: list) -> dict:
    """Kill rate of the first n fights of the shared order: exact for that order."""
    built = [r for r in recs if r["status"] in ("killed", "survived")]
    return {n: sum(1 for r in built if r["status"] == "killed" and r["first_kill_pos"] <= n) / max(1, len(built)) for n in ns}


__all__ = ["cmd_schemata", "guard", "kill_curve_shared", "math", "operand_left", "operand_right", "pick"]
