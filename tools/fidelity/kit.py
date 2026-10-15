"""Fix kits: what makes one backlog item cheap to fix, from the fix lab's groups (LAB/groups.json), inside its packet.

    python -m tools.fidelity.packet <item id>... --groups LAB/groups.json [--lab LAB]       kits for these items
    python -m tools.fidelity.packet --top N --groups LAB/groups.json --backlog <newest>      the top N open items + packets/grades.json
    python -m tools.fidelity.kit verify <Kit_*.cs>... [--emudump DLL]                        compile and run generated tests (the fixer's first step)

tools/fidelity/packet.py builds the kit when it is given --groups (without it, nothing here runs). A kit adds, for the
groups that match the item (the group's backlog_ids name it; else the group's implicated entities are ones the item names):

 1. minimal repros: each group example's game record, found under the lab (its ref: a path relative to LAB, optionally
    "#<seed>", or a seed searched in LAB/runs/*), replayed with this checkout's EmuDump: the first wrong step, the game's
    and the emulator's values there, the first differing event when the lab traced it (referee/trace_divergent.py);
 2. which ingredient causes it: the shrink's ingredient effect (the setup items without which the divergence vanishes);
 3. a test that FAILS on this checkout (test_source): the setup-fight pattern of tools/emudump.Tests (SetupMirrorTests: a
    referee record entered through the setup mirror, Fight.EnterRoom), carried to the first wrong step with the game's
    own actions and choices by Fight.Replay, and the game's state at each step as the expected values, compared the way
    tools/fidelity/fight_compare.py compares. src/Sts2Emulator.Tests cannot express a referee setup (the mirror lives in
    EmuDump), so the test is an EmuDump.Tests test. It is compiled against this checkout's EmuDump build in a scratch
    project, run, and kept next to the kit only if it fails at the repro's step and field (verify_tests);
 4. the game's rule pointer (rule_pointers): decompiled classes and methods, with paths and lines, of the implicated
    ingredients, the first event's hook, the item's game calls and the probe's game class. Names only, never code;
 5. siblings: the sweep table of the mechanism's hook or command (every listener and its emulator site) and the hook
    table rows (tools/fidelity/hook_table.py check) of that hook and of the implicated models;
 6. the probe answers (the packet's section 6).

Grade A = minimal repro + failing test verified + rule pointer + sibling list; B = minimal repro + rule pointer; C = the
rest. packets/grades.json holds {id: {grade, missing, main_sha}}. The generated tests hold the game's recorded actions
and values: they stay in the packet folder and are never committed. No game process is started here.
"""

from __future__ import annotations

import argparse
import contextlib
import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if __name__ == "__main__":
    sys.path.insert(0, str(ROOT))

from tools.fidelity import fight_compare as fc  # noqa: E402

BELOW_NORMAL = getattr(subprocess, "BELOW_NORMAL_PRIORITY_CLASS", 0)
PARTS = ("minimal repro", "failing test", "rule pointer", "sibling list", "first differing event", "ingredient effect", "probe answer")
GRADE_A = ("minimal repro", "failing test", "rule pointer", "sibling list")
GRADE_B = ("minimal repro", "rule pointer")
SEED = re.compile(r"^[0-9A-Z]{10}$")
# The record keys EmuDump's fight replay reads (tools/emudump/Fight.cs, Setup.cs); a test carries no others.
RECORD_KEYS = ("seed", "character", "ascension", "neow_pick", "deck_add", "room", "room_choices_from", "fight", "setup",
               "use_potions", "script", "choices")
MAX_EXAMPLES = 3  # examples replayed per group
MAX_GROUPS = 3  # groups per item that get a repro and a test
DOTNET_ENV = {"DOTNET_CLI_TELEMETRY_OPTOUT": "1", "DOTNET_NOLOGO": "1", "DOTNET_SKIP_FIRST_TIME_EXPERIENCE": "1"}


def clip(s, n: int = 300) -> str:
    s = s if isinstance(s, str) else json.dumps(s, separators=(",", ":"))
    s = " ".join(str(s).split())
    return s if len(s) <= n else s[: n - 3] + "..."


def read_json(path: Path, default=None):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return default


# ================================================================== groups

def load_groups(path: Path) -> dict:
    doc = read_json(path)
    if not isinstance(doc, dict) or not isinstance(doc.get("groups"), list):
        raise SystemExit(f"{path}: not a groups file (no 'groups' list)")
    for g in doc["groups"]:
        g.setdefault("examples", [])
        g.setdefault("entities", [])
        g.setdefault("ingredient_effect", [])
        g.setdefault("backlog_ids", [])
        g.setdefault("key", {})
    return doc


def setup_items(xs) -> list:
    """Setup items as (cmd, id) pairs, from spec dicts ({"cmd", "id"}) or strings ("relic STONE_CRACKER")."""
    out = []
    for x in xs or []:
        if isinstance(x, dict):
            out.append((str(x.get("cmd", "")), str(x.get("id") if x.get("id") is not None else x.get("amount", ""))))
        elif isinstance(x, str):
            cmd, _, rest = x.strip().partition(" ")
            out.append((cmd, rest.strip()))
    return out


def culprits(group: dict) -> list:
    """The setup items without which the divergence vanishes (ingredient_effect), as their ids."""
    out = []
    for e in group.get("ingredient_effect", []):
        if e.get("removal_vanishes"):
            item = e.get("item")
            pair = setup_items([item])[0] if isinstance(item, (dict, str)) and setup_items([item]) else ("", str(item))
            out.append(pair[1] or pair[0])
    return out


def group_entities(group: dict) -> list:
    ents = [str(e) for e in group.get("entities", [])]
    for c in culprits(group):
        if c not in ents:
            ents.append(c)
    return ents


def matching_groups(item: dict, item_entities: set, doc: dict) -> list:
    """[(group, how)] for one item: the groups whose backlog_ids name it, else the groups whose implicated entities the item
    names (two shared, or one when the group has only one); biggest impact first.
    """
    iid = item["id"]
    named = [(g, "backlog id") for g in doc["groups"] if iid in g.get("backlog_ids", [])]
    if named:
        return sorted(named, key=lambda t: (-(t[0].get("impact_runs_est") or 0), -(t[0].get("count") or 0), t[0]["id"]))
    out = []
    for g in doc["groups"]:
        ents = set(group_entities(g))
        shared = ents & set(item_entities)
        if shared and (len(shared) >= 2 or len(ents) == 1):
            out.append((g, "entities " + ", ".join(sorted(shared))))
    return sorted(out, key=lambda t: (-(t[0].get("impact_runs_est") or 0), -(t[0].get("count") or 0), t[0]["id"]))


# ================================================================== the minimal repro's game record

# EmuDump's own outputs beside a batch (<b>.emu.json, <b>.shrink.emu.json, <b>.reanchor.json): replays, never the game's record.
EMU_OUTPUTS = (".emu.json", ".reanchor.json")
# What fight_compare.game_view reads of a state in combat; EmuDump's states hold a snapshot and a draw order instead.
GAME_STATE_KEYS = ("player", "hand", "draw", "discard", "exhaust", "enemies", "rng")
EMU_STATE_KEYS = ("snapshot", "draw_order")
SHRINK_PROBE = re.compile(r"\.shrink\d+\.out\.json$")  # referee/shrink.py's rounds: the game's records of reduced setups


def is_game_record(r) -> bool:
    """Whether r is a game recording (the referee's shape: the fight's opening state and every state in combat hold what
    game_view reads), not EmuDump's replay of one or anything else that holds steps."""
    steps = r.get("steps") if isinstance(r, dict) else None
    if not isinstance(steps, list) or not steps:
        return False
    for s in steps:
        st = s.get("state") if isinstance(s, dict) else None
        if not isinstance(st, dict) or any(k in st for k in EMU_STATE_KEYS):
            return False
        if st.get("in_combat") and not all(k in st for k in GAME_STATE_KEYS):
            return False
    return bool(steps[0]["state"].get("in_combat"))


class RecordIndex:
    """The game records an example's ref names, read once.

    A ref is a path (optionally '#seed' or '#index') relative to the lab, or to a base its tag names ('referee/...': the
    referee folder, 'fork/...': the fork; tools/fidelity/lab_groups.py writes refs this way); a path without its extension is a
    batch (<b>.out.json and its shrink rounds <b>.shrinkN.out.json); a pack spec's record is found by its seed and setup. A
    bare seed is searched in every game record under LAB/runs. Only game records count (is_game_record): EmuDump's outputs
    beside them are skipped. A path that leaves its base is refused.
    """

    def __init__(self, lab: Path | None, bases: dict | None = None):
        self.lab = Path(lab) if lab else None
        self.bases = {k: Path(v) for k, v in (bases or {}).items() if v and k}
        if self.lab:
            self.bases[""] = self.lab
        self._by_seed: dict | None = None
        self._docs: dict = {}

    def doc(self, path: Path):
        key = str(Path(path).resolve())
        if key not in self._docs:
            if len(self._docs) > 8:
                self._docs.clear()
            self._docs[key] = read_json(path)
        return self._docs[key]

    @staticmethod
    def _holds_results(p: Path) -> bool:
        if p.name == "job.json" or p.name.endswith(EMU_OUTPUTS):
            return False
        try:
            if p.stat().st_size < 64:
                return False
            with open(p, encoding="utf-8", errors="replace") as f:
                return '"results"' in f.read(64)
        except OSError:
            return False

    def _records(self, p: Path, scan: bool = True) -> list:
        """[(path, index, record)] of the game records in one results file (scan: skip, unread, a file that is none)."""
        doc = self.doc(p) if not scan or self._holds_results(p) else None
        res = doc.get("results") if isinstance(doc, dict) else None
        return [(p, i, r) for i, r in enumerate(res or []) if is_game_record(r)]

    @property
    def by_seed(self) -> dict:
        if self._by_seed is None:
            self._by_seed = {}
            runs = self.lab / "runs" if self.lab else None
            if runs and runs.is_dir():
                for p in sorted(runs.rglob("*.json")):
                    for _, i, r in self._records(p):
                        if r.get("seed"):
                            self._by_seed.setdefault(r["seed"], []).append((p, i))
                    self._docs.clear()
        return self._by_seed

    def resolve(self, path: str) -> tuple[Path, str] | None:
        """(file system path, its base's tag) of a ref's path, or None when it is not a path under a known base."""
        if not path or ":" in path or "\\" in path or path.startswith("/"):
            return None  # refset:, corpus:, a codex line (<file>:<n>), an absolute path
        for tag in sorted(self.bases, key=len, reverse=True):
            if path.startswith(tag):
                base, rest = self.bases[tag], path[len(tag):]
                break
        else:
            return None
        if not rest or ".." in Path(rest).parts:
            return None
        p = base / rest
        try:
            p.resolve().relative_to(base.resolve())
        except (ValueError, OSError):
            return None
        return p, tag

    def label(self, p: Path) -> str:
        """A record file's path as refs write it: relative to the lab, or to a tagged base with its tag."""
        for tag in sorted(self.bases, key=len):
            with contextlib.suppress(ValueError, OSError):
                return tag + p.resolve().relative_to(self.bases[tag].resolve()).as_posix()
        return p.name

    def find(self, ref, minimal: list, seed_hint: str | None = None) -> tuple[dict | None, str]:
        """(record, where) for one example; `where` is the record file as label() writes it, '#<index>' added."""
        want = setup_items(minimal)
        ref = str(ref or "")
        path, _, sel = ref.partition("#")
        cands: list = []
        named = False  # the ref names a file or a batch that holds game records: they are the only candidates
        hit = None if SEED.match(path) else self.resolve(path)
        if hit and hit[0].is_file():
            p = hit[0]
            doc = self.doc(p) or {}
            if isinstance(doc, dict) and isinstance(doc.get("results"), list):
                cands += self._records(p, scan=False)
            elif isinstance(doc, dict) and isinstance(doc.get("runs"), list):  # a pack spec: its record is elsewhere
                spec = doc["runs"][0] if doc["runs"] else {}
                seed_hint = spec.get("seed") or seed_hint
                want = want or setup_items(spec.get("setup"))
            elif is_game_record(doc):
                cands.append((p, 0, doc))
            named = bool(cands)
        elif hit and hit[0].parent.is_dir():  # a batch: its records and its shrink rounds' records
            prefix = hit[0]
            for p in sorted(prefix.parent.glob(prefix.name + ".*json")):
                if p.name == prefix.name + ".out.json" or SHRINK_PROBE.search(p.name):
                    recs = self._records(p)
                    cands += recs
                    named = named or bool(recs)
        seed = sel if SEED.match(sel or "") else (path if SEED.match(path) else seed_hint)
        # a seed alone is searched lab-wide; a ref whose path held no game record only with a setup to tell its fight from
        # another fight on the same seed (a targets batch reuses recorded fights' seeds)
        if not cands and seed and not named and (not hit or want):
            for p, i in self.by_seed.get(seed, []):
                doc = self.doc(p) or {}
                cands.append((p, i, doc["results"][i]))
        if sel.isdigit():
            cands = [c for c in cands if c[1] == int(sel)]
        if seed:
            cands = [c for c in cands if c[2].get("seed") == seed]
        if want:
            exact = [c for c in cands if setup_items(c[2].get("setup")) == want]
            cands = exact or [c for c in cands if set(want) <= set(setup_items(c[2].get("setup")))]
            cands.sort(key=lambda c: (len(c[2].get("setup") or []), len(c[2].get("steps") or [])))
        else:  # no minimal setup to match: the fight the batch played, not a shrink round's reduced setup
            cands.sort(key=lambda c: bool(SHRINK_PROBE.search(c[0].name)))
        if not cands:
            return None, f"no game record found for ref {ref or '-'}" + (f" (seed {seed})" if seed else "")
        p, i, r = cands[0]
        return r, f"{self.label(p)}#{i}"


def where_text(where: str) -> str:
    """A record's place for the packet: LAB/<path> for the lab's, the tagged path for the others."""
    return where if where.startswith(("referee/", "fork/")) else "LAB/" + where


def trim_record(rec: dict, upto: int) -> dict:
    """What a test needs of a game record to replay it to step `upto`: the keys the replay reads, the game's own log of the
    setup items (ok/msg only), and the actions of steps 0..upto. The game's states are not carried (the test has them as
    its expected values).
    """
    out = {k: rec[k] for k in RECORD_KEYS if k in rec}
    if "setup_log" in rec:
        out["setup_log"] = [{k: e.get(k) for k in ("i", "cmd", "id", "ok", "msg") if k in e} for e in rec["setup_log"] if isinstance(e, dict)]
    out["steps"] = [{"action": s.get("action")} for s in (rec.get("steps") or [])[: upto + 1]]
    return out


def first_wrong(rec: dict, emu_res: dict, nm: dict):
    """(step, diffs, action) of the first step the emulator's replay does not reproduce, or None."""
    fd = fc.first_difference(rec, emu_res, nm)
    if fd is None:
        return None
    i, d = fd
    steps = rec.get("steps") or []
    return i, d, steps[i].get("action") if i < len(steps) else None


def action_label(a) -> str:
    if not a:
        return "the fight's start"
    kind = a.get("kind")
    if kind == "end_turn":
        return "END_TURN"
    if kind == "potion":
        return f"potion slot {a.get('slot')}" + (f" {a['potion']}" if a.get("potion") else "")
    return str(a.get("card") or a.get("card_id") or kind)


# ================================================================== the generated test

def cs_name(*parts) -> str:
    return re.sub(r"[^A-Za-z0-9_]", "_", "_".join(str(p) for p in parts if p))


def _raw(s: str) -> str:
    """A C# raw string literal holding s on one line (more quotes than any run inside it)."""
    n = max(3, max((len(m) for m in re.findall(r'"+', s)), default=0) + 1)
    q = '"' * n
    if s.startswith('"') or s.endswith('"'):
        s = " " + s + " "
    return f"{q}{s}{q}"


def test_source(cls: str, meta: dict, rec: dict, step: int, path: str) -> str:
    """One xunit test (tools/emudump.Tests' setup-fight style) that replays the minimal repro's record and expects the game's
    state at every step up to `step`; on the checkout it was made for it fails with 'KIT step <step> <path>:'.
    """
    trimmed = json.dumps(trim_record(rec, step), separators=(",", ":"), sort_keys=True)
    views = [json.dumps(fc.game_view(s["state"]), separators=(",", ":"), sort_keys=True) for s in rec["steps"][: step + 1]]
    renames = ", ".join(f'["{k}"] = "{v}"' for k, v in sorted(fc.EMU_RENAME.items()))
    not_powers = ", ".join(f'"{x}"' for x in sorted(fc.NOT_POWERS))
    intents = ", ".join(f'[{k}] = "{v}"' for k, v in sorted(fc.EMU_INTENT.items()))
    streams = ", ".join(f'"{x}"' for x in fc.RNG_STREAMS)
    optional = ", ".join(f'"{x}"' for x in sorted(fc.OPTIONAL))
    game_rows = ",\n".join(f"        {_raw(v)}" for v in views)
    head = "\n".join("// " + ln for ln in [
        f"Fix kit test for {meta.get('item')}: {clip(meta.get('title'), 150)}",
        f"Group {meta.get('group')}; minimal repro {rec.get('seed')} ({meta.get('setup') or 'the fight alone'}), game record {meta.get('where')}.",
        f"Generated by tools/fidelity/kit.py at fork {meta.get('sha', '?')}. NEVER COMMIT IT: it holds the game's recorded actions and values.",
        "It replays the game's record of the minimal setup as EmuDump fight does (Fight.Replay: the setup mirror, the game's",
        "actions and choices; the SetupMirrorTests pattern) and compares every step up to the first wrong one with the game's",
        "state, as tools/fidelity/fight_compare.py compares them (game_view against emu_view).",
        f"Expected on that fork: fails with 'KIT step {step} {path}:'. With the fix it passes.",
        "Run it: python -m tools.fidelity.kit verify <this file> (after rebuilding tools/emudump in Release).",
    ])
    return f"""{head}
using System.Globalization;
using System.Text;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using Xunit;

public class {cls}
{{
    const string Record = {_raw(trimmed)};

    /// <summary>The game's state at steps 0..{step}, as fight_compare.game_view gives it.</summary>
    static readonly string[] Game = new[]
    {{
{game_rows}
    }};

    [Fact]
    public void TheEmulatorReproducesTheGameUpToStep{step}()
    {{
        var outp = Fight.Replay(JsonNode.Parse(Record)!.AsObject());
        var steps = outp["steps"]!.AsArray();
        for (int i = 0; i < Game.Length; i++)
        {{
            if (i >= steps.Count)
            {{
                string why = outp["error"] != null ? "(error)" : "(length)";
                Assert.Fail($"KIT step {{i}} {{why}}: the emulator's replay has {{steps.Count}} steps; {{View.Clip(outp["error"]?.ToString() ?? "", 300)}}");
            }}

            var d = new List<(string Path, JsonNode? Game, JsonNode? Emu)>();
            View.Diff(JsonNode.Parse(Game[i]), View.Of(steps[i]!["state"]!.AsObject()), "", d);
            if (d.Count > 0)
            {{
                Assert.Fail($"KIT step {{i}} {{d[0].Path}}: game {{View.Clip(View.Canon(d[0].Game), 400)}} | emulator {{View.Clip(View.Canon(d[0].Emu), 400)}}"
                            + (d.Count > 1 ? $" (+{{d.Count - 1}} more: {{string.Join(", ", d.Skip(1).Select(x => x.Path))}})" : ""));
            }}
        }}
    }}

    /// <summary>tools/fidelity/fight_compare.py's emu_view and diff, for this test (the tables as they were at generation).</summary>
    static class View
    {{
        static readonly Dictionary<string, string> Rename = new() {{ {renames} }};
        static readonly HashSet<string> NotPowers = new() {{ {not_powers} }};
        static readonly Dictionary<int, string> Intents = new() {{ {intents} }};
        static readonly string[] Streams = {{ {streams} }};
        static readonly HashSet<string> Optional = new() {{ {optional} }};
        static readonly JsonObject IdMap = JsonNode.Parse(File.ReadAllText(Path.Combine(AppContext.BaseDirectory, "id_map.json")))!.AsObject();
        static readonly Dictionary<int, string> Cards = Names("cards"), Enemies = Names("enemies");

        static Dictionary<int, string> Names(string kind)
        {{
            var m = new Dictionary<int, string>();
            foreach (var kv in IdMap[kind]!.AsObject()) m[int.Parse(kv.Value!.ToString(), CultureInfo.InvariantCulture)] = Snake(kv.Key);
            return m;
        }}

        static string Snake(string n) => Regex.Replace(n, "(?<=[a-z0-9])(?=[A-Z])|(?<=[A-Z])(?=[A-Z][a-z])", "_").ToUpperInvariant();

        static string Power(string n)
        {{
            var s = Snake(n);
            return s.EndsWith("_POWER") ? s[..^"_POWER".Length] : s;
        }}

        static JsonNode Name(Dictionary<int, string> t, JsonNode? id)
        {{
            int k = int.Parse(id!.ToString(), CultureInfo.InvariantCulture);
            return t.TryGetValue(k, out var n) ? JsonValue.Create(n)! : JsonValue.Create(k)!;
        }}

        static bool Bool(JsonNode? n) => n is JsonValue v && (v.TryGetValue<bool>(out var b) ? b : Num(n) != 0);

        static double Num(JsonNode? n) => double.Parse(n!.ToJsonString(), CultureInfo.InvariantCulture);

        static JsonArray Card(JsonNode? c) => new(Name(Cards, c!["def_id"]), JsonValue.Create(Bool(c["upgraded"])));

        static JsonArray Powers(JsonArray? buffs)
        {{
            var l = new List<(string Name, JsonNode Amount)>();
            foreach (var b in buffs ?? new JsonArray())
            {{
                var p = Power((string)b!["name"]!);
                if (NotPowers.Contains(p)) continue;
                l.Add((Rename.TryGetValue(p, out var r) ? r : p, b["amount"]!.DeepClone()));
            }}

            return new JsonArray(l.OrderBy(x => x.Name, StringComparer.Ordinal).ThenBy(x => Num(x.Amount))
                .Select(x => (JsonNode)new JsonArray(JsonValue.Create(x.Name), x.Amount)).ToArray());
        }}

        static JsonNode? Intent(JsonObject e)
        {{
            if (e["intent"] is not JsonObject it) return null;
            if ((e["buffs"]?.AsArray() ?? new JsonArray()).Any(b => Power((string)b!["name"]!) == "STUNNED"))
                return new JsonObject {{ ["kind"] = "STUN", ["attack"] = null }};
            var sec = it["secondary"] as JsonObject;
            bool Carries(JsonObject o) => o["carries_damage"] is JsonNode c && Bool(c);
            JsonNode? attack = null;
            if ((int)Num(it["type"]) == 0 || Carries(it)) attack = new JsonArray(it["magnitude"]!.DeepClone(), it["hits"]!.DeepClone());
            else if (sec != null && ((int)Num(sec["type"]) == 0 || Carries(sec))) attack = new JsonArray(sec["magnitude"]!.DeepClone(), sec["hits"]!.DeepClone());
            string kind = Carries(it) ? "ATTACK" : Intents.TryGetValue((int)Num(it["type"]), out var k) ? k : "OTHER";
            return new JsonObject {{ ["kind"] = kind, ["attack"] = attack }};
        }}

        internal static JsonObject Of(JsonObject st)
        {{
            if (!Bool(st["in_combat"])) return End(st);
            if (st["snapshot"] is not JsonObject s) return new JsonObject {{ ["in_combat"] = false }};
            var p = s["player"]!;
            var piles = s["piles"]!;
            var v = new JsonObject
            {{
                ["in_combat"] = true,
                ["player"] = new JsonObject
                {{
                    ["hp"] = p["hp"]!.DeepClone(), ["max_hp"] = p["max_hp"]!.DeepClone(), ["block"] = p["block"]!.DeepClone(),
                    ["energy"] = p["energy"]!.DeepClone(), ["powers"] = Powers(p["buffs"]?.AsArray()),
                }},
                ["hand"] = new JsonArray(piles["hand"]!.AsArray().Select(c => (JsonNode)Card(c)).ToArray()),
                ["draw"] = new JsonArray(st["draw_order"]!.AsArray().Select(d => (JsonNode)new JsonArray(Name(Cards, d![0]), JsonValue.Create(Bool(d[1])))).ToArray()),
                ["discard"] = new JsonArray(piles["discard"]!.AsArray().Select(c => (JsonNode)Card(c)).ToArray()),
                ["exhaust"] = new JsonArray(piles["exhaust"]!.AsArray().Select(c => (JsonNode)Card(c)).ToArray()),
                ["enemies"] = new JsonArray(s["enemies"]!.AsArray().Where(e => Num(e!["hp"]) > 0).Select(e => (JsonNode)new JsonObject
                {{
                    ["name"] = Name(Enemies, e!["def_id"]), ["hp"] = e["hp"]!.DeepClone(), ["block"] = e["block"]!.DeepClone(),
                    ["powers"] = Powers(e["buffs"]?.AsArray()), ["intent"] = Intent(e.AsObject()),
                }}).ToArray()),
                ["rng"] = st["rng"]!.DeepClone(),
            }};
            if (st["potions"] is JsonNode pot) v["potions"] = pot.DeepClone();
            return v;
        }}

        static JsonObject End(JsonObject st)
        {{
            if (st["final"] is not JsonObject fin) return new JsonObject {{ ["in_combat"] = false }};
            if (!Bool(fin["won"])) return new JsonObject {{ ["in_combat"] = false, ["outcome"] = "lost" }};
            var p = fin["player"]!;
            var player = new JsonObject {{ ["hp"] = p["hp"]!.DeepClone(), ["max_hp"] = p["max_hp"]!.DeepClone(), ["block"] = 0, ["powers"] = new JsonArray() }};
            if (p["gold"] is JsonNode g) player["gold"] = g.DeepClone();
            var rng = new JsonObject();
            foreach (var kv in fin["rng"]!.AsObject()) if (Streams.Contains(kv.Key)) rng[kv.Key] = kv.Value!.DeepClone();
            var v = new JsonObject
            {{
                ["in_combat"] = false, ["outcome"] = "won", ["player"] = player, ["hand"] = new JsonArray(), ["draw"] = new JsonArray(),
                ["discard"] = new JsonArray(), ["exhaust"] = new JsonArray(), ["enemies"] = new JsonArray(), ["rng"] = rng,
            }};
            foreach (var k in new[] {{ "potions", "relics", "relic_counters" }}) if (fin[k] is JsonNode x) v[k] = x.DeepClone();
            return v;
        }}

        internal static void Diff(JsonNode? a, JsonNode? b, string path, List<(string, JsonNode?, JsonNode?)> outp)
        {{
            if (a is JsonObject oa && b is JsonObject ob)
            {{
                if (oa.ContainsKey("in_combat") && ob.ContainsKey("in_combat") && Bool(oa["in_combat"]) != Bool(ob["in_combat"]))
                {{
                    if (!Bool(oa["in_combat"])) oa = new JsonObject {{ ["in_combat"] = false }};
                    if (!Bool(ob["in_combat"])) ob = new JsonObject {{ ["in_combat"] = false }};
                }}

                if (oa.ContainsKey("outcome") && ob.ContainsKey("outcome") && Canon(oa["outcome"]) != Canon(ob["outcome"]))
                {{
                    outp.Add((path + ".outcome", oa["outcome"], ob["outcome"]));
                    return;
                }}

                foreach (var k in oa.Select(x => x.Key).Union(ob.Select(x => x.Key)).OrderBy(x => x, StringComparer.Ordinal))
                {{
                    string p = path + "." + k;
                    if (Optional.Contains(p) && !(oa.ContainsKey(k) && ob.ContainsKey(k))) continue;
                    Diff(oa[k], ob[k], p, outp);
                }}

                return;
            }}

            (a, b) = (Recorded(a, b), Recorded(b, a));
            if (Canon(a) != Canon(b)) outp.Add((path, a, b));
        }}

        /// <summary>fight_compare._recorded: enemy lists without the intents the other side did not record.</summary>
        static JsonNode? Recorded(JsonNode? xs, JsonNode? ys)
        {{
            if (xs is not JsonArray xa || ys is not JsonArray ya) return xs;
            if (!xa.Concat(ya).Any(x => x is JsonObject o && o.ContainsKey("intent"))) return xs;
            var r = new JsonArray();
            for (int i = 0; i < xa.Count; i++)
            {{
                var x = xa[i]?.DeepClone();
                bool other = i < ya.Count && ya[i] is JsonObject yo && yo["intent"] != null;
                if (x is JsonObject xo && (xo["intent"] == null || !other)) xo.Remove("intent");
                r.Add(x);
            }}

            return r;
        }}

        internal static string Canon(JsonNode? n)
        {{
            switch (n)
            {{
                case null: return "null";
                case JsonObject o: return "{{" + string.Join(",", o.OrderBy(kv => kv.Key, StringComparer.Ordinal).Select(kv => System.Text.Json.JsonSerializer.Serialize(kv.Key) + ":" + Canon(kv.Value))) + "}}";
                case JsonArray a: return "[" + string.Join(",", a.Select(Canon)) + "]";
                default:
                    var v = n.AsValue();
                    if (v.TryGetValue<bool>(out var b)) return b ? "true" : "false";
                    if (v.TryGetValue<string>(out var s)) return System.Text.Json.JsonSerializer.Serialize(s);
                    double d = Num(n);
                    return d == Math.Floor(d) && Math.Abs(d) < 1e15 ? ((long)d).ToString(CultureInfo.InvariantCulture) : d.ToString("R", CultureInfo.InvariantCulture);
            }}
        }}

        internal static string Clip(string s, int n) => s.Length <= n ? s : s[..(n - 3)] + "...";
    }}
}}
"""


# ================================================================== compiling and running the tests

CSPROJ = """<Project Sdk="Microsoft.NET.Sdk">
  <!-- Scratch project for fix-kit tests (tools/fidelity/kit.py): tools/emudump.Tests' settings, against a built EmuDump. -->
  <PropertyGroup>
    <TargetFramework>net9.0</TargetFramework>
    <Nullable>enable</Nullable>
    <ImplicitUsings>enable</ImplicitUsings>
    <IsPackable>false</IsPackable>
    <AssemblyName>EmuDump.Tests</AssemblyName>
    <NuGetAudit>false</NuGetAudit>
    <TreatWarningsAsErrors>false</TreatWarningsAsErrors>
    <NoWarn>$(NoWarn);CS8600;CS8602;CS8604;CS8619</NoWarn>
  </PropertyGroup>
  <ItemGroup>
    <PackageReference Include="Microsoft.NET.Test.Sdk" Version="{test_sdk}" />
    <PackageReference Include="xunit" Version="{xunit}" />
    <PackageReference Include="xunit.runner.visualstudio" Version="{runner}" />
  </ItemGroup>
  <ItemGroup>
    <Reference Include="EmuDump"><HintPath>{bin}/EmuDump.dll</HintPath></Reference>
    <Reference Include="Sts2Emulator"><HintPath>{bin}/Sts2Emulator.dll</HintPath></Reference>
    <None Include="{bin}/id_map.json" Link="id_map.json" CopyToOutputDirectory="PreserveNewest" />
    <None Include="{bin}/power_aliases.json" Link="power_aliases.json" CopyToOutputDirectory="PreserveNewest" />
  </ItemGroup>
</Project>
"""


def package_versions(tests_proj: Path) -> dict:
    """The xunit package versions tools/emudump.Tests uses (so the offline restore finds them in the local cache)."""
    text = tests_proj.read_text(encoding="utf-8") if tests_proj.is_file() else ""

    def v(name, default):
        m = re.search(r'Include="' + re.escape(name) + r'"\s+Version="([^"]+)"', text)
        return m.group(1) if m else default
    return {"test_sdk": v("Microsoft.NET.Test.Sdk", "17.10.0"), "xunit": v("xunit", "2.9.0"), "runner": v("xunit.runner.visualstudio", "2.8.2")}


def _run(cmd: list, cwd: Path, timeout: float, log: Path) -> int:
    env = {**os.environ, **DOTNET_ENV}
    with open(log, "a", encoding="utf-8") as f:
        f.write("$ " + " ".join(str(c) for c in cmd) + "\n")
        f.flush()
        try:
            p = subprocess.run([str(c) for c in cmd], cwd=cwd, stdout=f, stderr=subprocess.STDOUT, env=env, check=False,
                               creationflags=BELOW_NORMAL, timeout=timeout)
            return p.returncode
        except subprocess.TimeoutExpired:
            f.write(f"timed out after {timeout:.0f} s\n")
            return 124


def parse_trx(path: Path) -> dict:
    """{test class: (outcome, message)} from a dotnet test TRX log."""
    out = {}
    if not path.is_file():
        return out
    root = ET.parse(path).getroot()
    for el in root.iter():
        if el.tag.rsplit("}", 1)[-1] != "UnitTestResult":
            continue
        name = el.get("testName", "")
        msg = ""
        for sub in el.iter():
            if sub.tag.rsplit("}", 1)[-1] == "Message" and sub.text:
                msg = sub.text
                break
        out[name.split(".")[0]] = (el.get("outcome", "?"), msg.strip())
    return out


COMPILE_ERR = re.compile(r"([^\\/\s(]+\.cs)\(\d+,\d+\): error ([A-Z]+\d+): (.*?)(?: \[|$)")
ABS_PATH = re.compile(r"(?:[A-Za-z]:[\\/]|/[a-z]/|/(?:home|Users|tmp)/)[^\s\"'<>|]*")


def scrub(text: str) -> str:
    """A tool's output without the machine's folders (results and logs hold no user names or absolute paths): each absolute
    path becomes its last component."""
    return ABS_PATH.sub(lambda m: "<" + re.split(r"[\\/]", m.group().rstrip("\\/"))[-1] + ">", text or "")


def _log_tail(log: Path, mark: int = 0, n: int = 300) -> str:
    text = log.read_text(encoding="utf-8", errors="replace")[mark:] if log.is_file() else ""
    lines = [ln for ln in text.splitlines() if ln.strip() and not ln.startswith("$ ")]
    return clip(scrub(" | ".join(lines[-4:])), n)


def verify_tests(sources: dict, emudump: Path, work: Path, timeout: float = 600) -> dict:
    """Compile and run generated tests ({class name: C# source}) in one scratch project against an EmuDump build.

    Returns {class: {"outcome": failed|passed|compile_error|error, "message": str}}. A source that does not compile is
    dropped and the rest are built again, so one bad test does not cost the others.
    """
    res: dict = {}
    if not sources:
        return res
    bin_dir = Path(emudump).resolve().parent
    if not (bin_dir / "EmuDump.dll").is_file() or not shutil.which("dotnet"):
        return {k: {"outcome": "error", "message": "no EmuDump build or no dotnet"} for k in sources}
    work = Path(work)
    proj = work / "kit-tests"
    if proj.exists():
        shutil.rmtree(proj)
    proj.mkdir(parents=True)
    log = work / "kit-tests.log"
    vers = package_versions(ROOT / "tools" / "emudump.Tests" / "EmuDump.Tests.csproj")
    (proj / "KitTests.csproj").write_text(CSPROJ.format(bin=bin_dir.as_posix(), **vers), encoding="utf-8")
    (proj / "AssemblyInfo.cs").write_text("using Xunit;\n\n[assembly: CollectionBehavior(DisableTestParallelization = true)]\n", encoding="utf-8")
    for name, text in sources.items():
        (proj / f"{name}.cs").write_text(text, encoding="utf-8")
    cache = Path.home() / ".nuget" / "packages"
    if _run(["dotnet", "restore", proj, "--source", cache, "-v", "q", "-p:NuGetAudit=false"], proj, timeout, log) != 0:
        return {k: {"outcome": "error", "message": "offline restore failed: " + _log_tail(log)} for k in sources}
    live = dict(sources)
    for _attempt in range(3):
        mark = log.stat().st_size if log.is_file() else 0
        rc = _run(["dotnet", "build", proj, "--no-restore", "-v", "q", "-nologo"], proj, timeout, log)
        if rc == 0:
            break
        text = log.read_text(encoding="utf-8", errors="replace")[mark:]
        bad = {}
        for m in COMPILE_ERR.finditer(text):
            stem = m.group(1)[:-3]
            if stem in live and stem not in bad:
                bad[stem] = f"{m.group(2)}: {m.group(3)}"
        if not bad:
            return {**res, **{k: {"outcome": "error", "message": "the build failed: " + _log_tail(log, mark)} for k in live}}
        for stem, why in bad.items():
            res[stem] = {"outcome": "compile_error", "message": scrub(why)}
            live.pop(stem)
            (proj / f"{stem}.cs").unlink()
        if not live:
            return res
    else:
        return {**res, **{k: {"outcome": "error", "message": "the build kept failing"} for k in live}}
    trx = work / "kit-tests.trx"
    trx.unlink(missing_ok=True)
    _run(["dotnet", "test", proj, "--no-build", "--logger", f"trx;LogFileName={trx.name}", "--results-directory", work], proj, timeout, log)
    got = parse_trx(trx)
    for name in live:
        outcome, msg = got.get(name, ("error", "no result in the test log"))
        res[name] = {"outcome": {"Failed": "failed", "Passed": "passed"}.get(outcome, "error"), "message": scrub(msg)}
    return res


def fails_as_stated(result: dict, step: int, path: str) -> bool:
    return result.get("outcome") == "failed" and f"KIT step {step} {path}:" in result.get("message", "")


# ================================================================== rule pointers and siblings

PHASE_WORDS = {  # field (or the step's kind) -> words of the hook and method names that act on it
    ".draw": ("Draw", "Shuffle", "Hand", "RoomEntered", "CombatStart", "Upgrade"),
    ".hand": ("Draw", "Hand", "RoomEntered", "CombatStart", "TurnStart", "Upgrade", "Play"),
    ".discard": ("Play", "Discard", "Cost", "Energy", "Use"),
    ".exhaust": ("Exhaust", "Play"),
    ".player.energy": ("Energy", "CombatStart", "TurnStart", "RoomEntered"),
    ".player.hp": ("Damage", "HpLost", "Heal", "Attack"),
    ".player.block": ("Block", "Attack", "Damage"),
    ".player.powers": ("Power", "Apply", "TurnStart", "TurnEnd"),
    ".enemies": ("RoomEntered", "CombatStart", "Damage", "HpLost", "Block", "Power"),
    ".potions": ("Potion", "Procure", "RoomEntered", "CombatStart", "Use"),
    ".rng": ("Draw", "Shuffle", "Random", "Select", "RoomEntered", "CombatStart"),
}
FOLDERS = {"card": "Cards", "enemy": "Monsters", "power": "Powers", "relic": "Relics", "potion": "Potions"}


def pascal_of(key: str) -> str:
    return "".join(w.capitalize() for w in str(key).split("_") if w)


def method_line(text: str, method: str) -> int | None:
    m = re.search(r"\b(?:override|static|virtual|public|private|internal|protected|async)\b[^;{=\n]*\b" + re.escape(method)
                  + r"\s*(?:<[^>()]*>)?\s*\(", text)
    return text.count("\n", 0, m.start()) + 1 if m else None


def class_files(key: str, names, decomp) -> list:
    """Decompiled model class files of an entity key (a card, relic, power, potion, monster or encounter id)."""
    if not decomp.root:
        return []
    out = []
    for kind, pas in names.pascals.get(key, []):
        folder = FOLDERS.get(kind)
        out += [q for q in decomp.files.get(pas, []) if not folder or f"Models.{folder}" in str(q)]
    if not out:  # encounters and other models id_map does not list: by the Pascal form of the id
        out += [q for q in decomp.files.get(pascal_of(key), []) if "Models" in str(q)]
    return list(dict.fromkeys(out))


def relevance(name: str, field: str) -> int:
    key = next((k for k in PHASE_WORDS if field.startswith(k)), None)
    return sum(1 for w in PHASE_WORDS.get(key, ()) if w.lower() in name.lower())


def _pairs(implicated: list) -> list:
    """(key, why) pairs from implicated entities given as keys or as pairs."""
    return [(x, "implicated") if isinstance(x, str) else (str(x[0]), str(x[1])) for x in implicated]


MAIN = ("the shrink's culprit", "named by the group", "implicated")  # implicated entities whose own rule is the suspect


def rule_pointers(implicated: list, field: str, hook: str | None, calls: list, probe_classes: list, names, decomp,
                  rel_decomp, limit: int = 8) -> list:
    """[{where, what, why}] pointing at the decompiled classes and methods that hold the game's rule (names, paths, lines).

    implicated: entity keys, or (key, why) pairs, the shrink's culprits first; hook: the first differing event's hook, if
    known; calls: [(Class, Method, path)] the item's evidence names; probe_classes: the matched probes' game classes.
    """
    out, seen = [], set()

    def add(path: Path, method: str | None, why: str):
        k = (str(path), method)
        if k in seen:
            return
        seen.add(k)
        ln = method_line(decomp.text(path), method) if method else None
        out.append({"where": rel_decomp(path) + (f":{ln}" if ln else ""), "what": f"{path.stem}.{method}" if method else path.stem, "why": why})

    for key, why in _pairs(implicated):
        n = 3 if why in MAIN else 1
        for p in class_files(key, names, decomp)[:2]:
            ovs = decomp.overrides(p)
            if hook and hook in ovs:
                add(p, hook, f"{key} ({why}) overrides the first differing event's hook")
            ranked = sorted(ovs, key=lambda m: (-relevance(m, field), ovs.index(m)))
            for m in ranked[:n]:
                add(p, m, f"{key} ({why}); it overrides {m}" + (" (fits the differing field)" if relevance(m, field) else ""))
            if not ovs:
                add(p, None, f"{key} ({why}); no overrides: its rule is in its data or a caller")
    if hook:
        for stem in ("Hook", "AbstractModel"):
            for p in decomp.files.get(stem, []):
                if decomp.declared(p, hook):
                    add(p, hook, "the first differing event's hook: its declaration and dispatch")
    for _cls, meth, p in calls:
        add(Path(p), meth, "named by the item's evidence")
    for gc in probe_classes:
        for tok in re.findall(r"\b([A-Z][A-Za-z0-9]+)(?:\.([A-Z][A-Za-z0-9]+))?", gc or ""):
            for p in decomp.files.get(tok[0], [])[:1]:
                add(p, tok[1] or None, "the rule probe's game class")
    return out[:limit]


def method_body(text: str, method: str) -> str:
    """The text of one method (its declaration to its closing brace, or an expression body to its semicolon); '' if absent."""
    m = re.search(r"\b(?:override|static|virtual|public|private|internal|protected|async)\b[^;{=\n]*\b" + re.escape(method)
                  + r"\s*(?:<[^>()]*>)?\s*\(", text)
    if not m:
        return ""
    i = m.end()
    brace, arrow = text.find("{", i), text.find("=>", i)
    semi = text.find(";", i)
    if arrow != -1 and (brace == -1 or arrow < brace) and (semi == -1 or arrow < semi):
        return text[m.start(): text.find(";", arrow) + 1]
    if brace == -1:
        return ""
    depth, j = 0, brace
    while j < len(text):
        c = text[j]
        if c == "{":
            depth += 1
        elif c == "}":
            depth -= 1
            if depth == 0:
                return text[m.start(): j + 1]
        j += 1
    return text[m.start():]


SKIP_QUAL = {"Task", "Math", "System", "Core", "MegaCrit", "Sts2", "TestMode", "HoverTipFactory", "Log", "Enumerable", "Owner",
             "Mathf", "Callable", "GD", "StringComparison", "LocString"}


# Instance calls that are plumbing, not a game rule: collections and LINQ, tasks, checks, UI.
SKIP_CALLS = {"Add", "AddRange", "Remove", "RemoveAt", "RemoveAll", "Contains", "ContainsKey", "Where", "Select", "SelectMany",
              "Any", "All", "First", "FirstOrDefault", "Last", "LastOrDefault", "Single", "SingleOrDefault", "Count", "ToList",
              "ToArray", "ToHashSet", "ToDictionary", "OrderBy", "OrderByDescending", "ThenBy", "Take", "Skip", "Sum", "Max", "Min",
              "Concat", "Distinct", "Except", "Intersect", "Cast", "OfType", "Reverse", "Sort", "Append", "Prepend", "Insert",
              "IndexOf", "TryGetValue", "Clear", "GetType", "ToString", "Equals", "GetHashCode", "Invoke", "Format", "Join",
              "Split", "Replace", "StartsWith", "EndsWith", "ConfigureAwait", "GetAwaiter", "GetResult", "WhenAll", "Delay",
              "FromResult", "Run", "FindOnTable", "CustomScaledWait", "Flash", "Preview", "GetPile"}
UI = re.compile(r"Preview|Vfx|Sfx|Anim|Tween|Hover|Wait|Flash|Shake|Sound|Music|Node|Scene|Layout|Display|Tooltip")  # presentation, no rule
_TEXT: dict = {}


def decompile_text(decomp) -> str:
    """The whole decompile as one text (about 18 MB), read once per process, to count a name's sites."""
    key = str(decomp.root)
    if key not in _TEXT:
        _TEXT[key] = "\n".join(decomp.text(p) for ps in decomp.files.values() for p in ps) if decomp.root else ""
    return _TEXT[key]


def site_count(decomp, target: str) -> int:
    """How often the decompile uses a Class.Member, or calls a bare method name."""
    pat = (r"\b" + re.escape(target) + r"\b") if "." in target else (r"\." + re.escape(target) + r"\s*(?:<[^>()]*>)?\s*\(")
    return len(re.findall(pat, decompile_text(decomp)))


def body_targets(decomp, path: Path, method: str) -> list:
    """Game identifiers a method's rule runs through, to sweep for siblings (names only): the commands it calls (Commands and
    Hooks namespaces), the RNG streams it draws on (Rng.X), the enum members it tests (RoomType.Elite) and the game methods it
    calls on objects (SetThisTurnOrUntilPlayed), plumbing left out."""
    body = method_body(decomp.text(path), method)
    found = []
    for cls, mem in re.findall(r"(?<![\w.])([A-Z][A-Za-z0-9]+)\.([A-Z][A-Za-z0-9]+)\b", body):
        if cls in SKIP_QUAL or cls == path.stem or UI.search(cls + "." + mem):
            continue
        files = decomp.files.get(cls, [])
        if cls == "Rng" or any("Commands" in str(f) or "Hooks" in str(f) for f in files) or (
                files and re.search(r"\benum\s+" + re.escape(cls) + r"\b", decomp.text(files[0]))):
            found.append(f"{cls}.{mem}")
    members = {t.split(".")[1] for t in found}
    for name in re.findall(r"(?<=[a-z0-9_)\]])\.([A-Z][A-Za-z0-9]+)\s*(?:<[^>()]*>)?\s*\(", body):
        if name not in SKIP_CALLS and name not in members and not name.startswith(("Assert", "Throw", "Log")) and not UI.search(name):
            found.append(name)
    return list(dict.fromkeys(found))


def sibling_targets(implicated: list, field: str, hook: str | None, names, decomp, hook_names: set, limit: int = 3) -> list:
    """What to sweep for sibling sites: the first event's hook, else the hook the implicated classes override that fits the
    differing field best; then the names their relevant overrides run through (body_targets) that the fewest game sites
    share (two at least: one is only itself), so the sweep lists the mechanism's users, not every caller of a common verb."""
    hooks, calls = ([hook] if hook else []), []
    for key, why in _pairs(implicated):
        if why not in MAIN:
            continue
        for p in class_files(key, names, decomp)[:1]:
            ovs = decomp.overrides(p)
            ranked = sorted(ovs, key=lambda m: (-relevance(m, field), ovs.index(m)))
            if not hook:
                hooks += [m for m in ranked if m in hook_names and relevance(m, field) and m not in hooks]
            for m in ranked[:2]:
                calls += [t for t in body_targets(decomp, p, m) if t not in calls and t not in hook_names]
    scored = sorted((n, i, t) for i, t in enumerate(calls) if (n := site_count(decomp, t)) >= 2)
    hooks = list(dict.fromkeys(hooks))[:1]
    return hooks + [t for _, _, t in scored][: limit - len(hooks)]


def hook_rows(check: dict | None, models: set, hooks: set, limit: int = 14) -> list:
    """Lines of the hook table (tools/fidelity/hook_table.py check) for the implicated models and the sibling hooks."""
    if not check or not check.get("rows"):
        return []
    rows = check["rows"]
    out = []
    mine = [r for r in rows if r["model"] in models]
    for r in mine[:limit]:
        out.append(f"  - {r['model']}.{r['hook']} ({r['kind']}): {r['status']}" + (f" at {', '.join(r['where'][:2])}" if r.get("where") else "")
                   + (f"; {clip(r['reason'], 120)}" if r.get("reason") else ""))
    for h in sorted(hooks):
        hr = [r for r in rows if r["hook"] == h]
        if not hr:
            continue
        counts = {}
        for r in hr:
            counts[r["status"]] = counts.get(r["status"], 0) + 1
        out.append(f"  - hook {h}: {len(hr)} listeners in the game ({', '.join(f'{k} {v}' for k, v in sorted(counts.items()))})")
        odd = [r for r in hr if r["status"] != "registered" and r["model"] not in models]
        for r in odd[:6]:
            out.append(f"    - {r['model']} ({r['kind']}): {r['status']}" + (f"; {clip(r['reason'], 100)}" if r.get("reason") else ""))
        sites = sorted({w.split(":")[0] for r in hr for w in r.get("where", [])})
        if sites:
            out.append(f"    - emulator dispatch points: {', '.join(sites[:6])}" + (" ..." if len(sites) > 6 else ""))
    return out


def load_hook_check(cache: Path, stamp: str, decompiled: Path | None, timeout: float = 240) -> dict | None:
    """The hook table check (about 25 s), cached per fork commit next to the sweep tables."""
    doc = read_json(cache)
    if isinstance(doc, dict) and doc.get("_stamp") == stamp and doc.get("rows"):
        return doc
    if not decompiled or not Path(decompiled).is_dir():
        return None
    cache.parent.mkdir(parents=True, exist_ok=True)
    tmp = cache.with_suffix(".tmp.json")
    with contextlib.suppress(Exception):
        subprocess.run([sys.executable, "-m", "tools.fidelity.hook_table", "check", "--decompiled", str(decompiled), "--json", str(tmp)],
                       cwd=ROOT, capture_output=True, check=False, timeout=timeout, creationflags=BELOW_NORMAL,
                       env={**os.environ, "PYTHONDONTWRITEBYTECODE": "1"})
    doc = read_json(tmp)
    if not isinstance(doc, dict) or not doc.get("rows"):
        return None
    doc["_stamp"] = stamp
    cache.write_text(json.dumps(doc), encoding="utf-8")
    tmp.unlink(missing_ok=True)
    return doc


# ================================================================== the kit

def grade(have: set) -> tuple[str, list]:
    g = "A" if all(p in have for p in GRADE_A) else "B" if all(p in have for p in GRADE_B) else "C"
    return g, [p for p in PARTS if p not in have]


def event_text(ev) -> str:
    if not ev:
        return ""
    if isinstance(ev, dict):
        return ", ".join(f"{k} {clip(v, 160)}" for k, v in ev.items() if v not in (None, "", []))
    return clip(ev, 400)


def event_hook(group: dict, ex: dict | None) -> str | None:
    key = group.get("key") or {}
    for k in ("hook", "event_hook"):
        if isinstance(key.get(k), str) and key[k]:
            return key[k]
    ev = (ex or {}).get("first_event")
    if isinstance(ev, dict) and isinstance(ev.get("hook"), str):
        return ev["hook"]
    if isinstance(ev, str):
        m = re.search(r"\b((?:After|Before|Modify|Should|On)[A-Z][A-Za-z]+)\b", ev)
        if m:
            return m.group(1)
    return None


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=(__doc__ or "").split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    v = sub.add_parser("verify", help="compile and run generated kit tests against an EmuDump build")
    v.add_argument("tests", nargs="+", type=Path)
    v.add_argument("--emudump", type=Path, default=ROOT / "tools/emudump/bin/Release/net9.0/EmuDump.dll")
    a = ap.parse_args(argv)
    sources = {}
    for t in a.tests:
        text = t.read_text(encoding="utf-8")
        m = re.search(r"^public class (\w+)", text, re.MULTILINE)
        if not m:
            print(f"{t}: no test class", file=sys.stderr)
            return 2
        sources[m.group(1)] = text
    with tempfile.TemporaryDirectory(prefix="kit-verify-") as tmp:
        res = verify_tests(sources, a.emudump, Path(tmp))
    rc = 0
    for name, r in res.items():
        print(f"{name}: {r['outcome'].upper()} {clip(r['message'], 600)}")
        rc = rc or (0 if r["outcome"] == "passed" else 1)
    return rc


if __name__ == "__main__":
    sys.exit(main())
