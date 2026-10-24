// EmuDump scenario <file|dir> [--engine old|core2] [--json] [--dump]: engine-neutral scenario tests (docs/rebuild/SCENARIOS.md).
// A scenario file (tests/scenarios/*.json) holds the setup in the game's terms (character, ascension, relics in obtain order, deck,
// potions by slot, powers with amounts on creatures by slot, the encounter, HP and energy), the actions (play a card by ModelId or
// hand index, a potion, end turn, screen answers by candidate ModelId) and the expected fields at named points as paths into the
// game view ({snapshot, net, private} of IEngine.GameView, plus `screen`). Each one is run through IEngine only: the setup becomes
// the setup items Setup.cs already runs (relics_set, deck_set, potions_set, hp, fight, power, energy, card), the actions are the
// engine's own PlayCard/UsePotion/EndTurn/Pick/Decline, and the expectations are read off the engine's GameView. No engine type
// appears here (tools/emudump.Tests/EngineBoundaryTests.cs).
//
// Outcome per scenario and engine: pass; fail (a difference, an engine error, an illegal action); xfail (a fail the scenario lists for
// this engine under `xfail_engines`, with its reason); xpass (listed as an xfail and passed: unexpected, take the row out); skip (the
// scenario's `engines` excludes this engine); invalid (the file breaks the format: always unexpected). Exit 1 on fail, xpass or invalid.
using System.Text.Json;
using System.Text.Json.Nodes;
using System.Text.RegularExpressions;
using EmuDump.Engine;
using EmuDump.Engines;

static class ScenarioMode
{
    internal static int Run(string[] args)
    {
        var paths = args.Where(a => !a.StartsWith("--", StringComparison.Ordinal)).ToList();
        if (paths.Count != 1 || args.Any(a => a.StartsWith("--", StringComparison.Ordinal) && a is not ("--json" or "--dump")))
        {
            Console.Error.WriteLine("usage: EmuDump scenario <file|dir> [--engine old|core2] [--json] [--dump]");
            return 2;
        }

        bool json = args.Contains("--json"), dump = args.Contains("--dump");
        string engine = EngineFactory.Selected;
        string root = paths[0];
        List<(string Name, string Path)> files;
        if (File.Exists(root))
        {
            files = [(System.IO.Path.GetFileName(root), root)];
        }
        else if (Directory.Exists(root))
        {
            files = Directory.EnumerateFiles(root, "*.json", SearchOption.AllDirectories)
                .Select(p => (System.IO.Path.GetRelativePath(root, p).Replace('\\', '/'), p))
                .OrderBy(f => f.Item1, StringComparer.Ordinal).ToList();
        }
        else
        {
            Console.Error.WriteLine($"EmuDump scenario: no such file or directory: {root}");
            return 2;
        }

        var results = new List<ScenarioResult>();
        foreach (var (name, path) in files)
        {
            JsonNode? doc;
            try
            {
                doc = JsonNode.Parse(File.ReadAllText(path));
            }
            catch (JsonException ex)
            {
                results.Add(ScenarioResult.Invalid(name, "", engine, "not JSON: " + ex.Message));
                continue;
            }

            var list = doc is JsonArray arr ? arr.ToList() : [doc];
            foreach (var sc in list)
            {
                results.Add(Scenarios.Run(name, sc, engine, dump));
            }
        }

        string summary = Summary(results);
        if (json)
        {
            var o = new JsonObject
            {
                ["engine"] = engine,
                ["summary"] = JsonNode.Parse(summary),
                ["results"] = new JsonArray(results.Select(r => (JsonNode)r.ToJson()).ToArray()),
            };
            Console.Out.WriteLine(o.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
        }
        else
        {
            foreach (var r in results)
            {
                Console.Out.WriteLine(r.Line());
                foreach (var d in r.Diffs) Console.Out.WriteLine("    " + d);
            }

            Console.Out.WriteLine($"scenarios on {engine}: {summary}");
        }

        return results.Any(r => r.Unexpected) ? 1 : 0;
    }

    static string Summary(List<ScenarioResult> results)
    {
        int Count(string s) => results.Count(r => r.Status == s);
        return new JsonObject
        {
            ["pass"] = Count("pass"), ["fail"] = Count("fail"), ["xfail"] = Count("xfail"), ["xpass"] = Count("xpass"),
            ["skip"] = Count("skip"), ["invalid"] = Count("invalid"), ["unexpected"] = results.Count(r => r.Unexpected),
        }.ToJsonString();
    }
}

sealed class ScenarioResult
{
    internal string File = "", Name = "", Engine = "", Status = "pass", Source = "";
    internal string? Reason;
    internal List<string> Diffs = [];
    internal JsonObject? Points;

    internal bool Unexpected => Status is "fail" or "xpass" or "invalid";

    internal static ScenarioResult Invalid(string file, string name, string engine, string why) =>
        new() { File = file, Name = name, Engine = engine, Status = "invalid", Diffs = [why] };

    internal string Line() => $"{Status.ToUpperInvariant(),-7}{File}: {Name}" + (Status == "xfail" ? $" (known: {Reason})" : "")
        + (Status == "xpass" ? $" (listed as failing on {Engine}: {Reason}; it passes, so drop the xfail_engines entry)" : "");

    internal JsonObject ToJson()
    {
        var o = new JsonObject
        {
            ["file"] = File, ["name"] = Name, ["engine"] = Engine, ["status"] = Status, ["unexpected"] = Unexpected,
            ["reason"] = Reason, ["diffs"] = new JsonArray(Diffs.Select(d => (JsonNode)JsonValue.Create(d)!).ToArray()),
        };
        if (Points != null) o["points"] = Points.DeepClone();
        return o;
    }
}

static class Scenarios
{
    static readonly string[] Top = ["name", "source", "doc", "engines", "xfail_engines", "setup", "start_answers", "actions", "expect"];
    static readonly string[] SetupKeys =
        ["character", "ascension", "seed", "encounter", "relics", "deck", "potions", "hp", "powers", "energy", "hand", "draw", "discard"];
    static readonly string[] ActionKeys = ["name", "do", "card", "hand_index", "upgraded", "slot", "target", "answers"];
    static readonly string[] ExpectKeys = ["at", "path", "equals", "one_of", "at_least", "at_most", "absent", "delta_from", "why"];
    static readonly string[] Operators = ["equals", "one_of", "at_least", "at_most", "absent"];
    static readonly Regex SourceForm = new(@"^(game|recording|ledger): \S", RegexOptions.Compiled);

    /// <summary>The act number of the `act` setup item for the game's acts (the first two share act 1).</summary>
    static int ActNumber(string act) => act switch { "OVERGROWTH" or "UNDERDOCKS" => 1, "HIVE" => 2, "GLORY" => 3, _ => 0 };

    // ---------------------------------------------------------------------------------------------------- format check

    /// <summary>What is wrong with a scenario (empty when it is well formed). Unknown keys are errors, so a typo cannot pass quietly.</summary>
    internal static List<string> Validate(JsonNode? node)
    {
        var p = new List<string>();
        if (node is not JsonObject sc) return ["a scenario is a JSON object"];
        CheckKeys(sc, "scenario", Top, ["name", "source", "setup", "expect"], p);
        if (sc["name"] is not JsonValue n || n.TryGetValue<string>(out var nm) is false || string.IsNullOrWhiteSpace(nm)) p.Add("name: a non-empty string");
        if (sc["source"] is not JsonValue s || !s.TryGetValue<string>(out var src) || !SourceForm.IsMatch(src)) p.Add("source: 'game: <Class>.<Method>', 'recording: <id>' or 'ledger: <id>'");
        var engines = new List<string>();
        if (sc["engines"] != null)
        {
            if (sc["engines"] is not JsonArray ea || ea.Count == 0 || ea.Any(x => x is not JsonValue v || !v.TryGetValue<string>(out var en) || !EngineFactory.Known.Contains(en)))
                p.Add("engines: a non-empty list of " + string.Join("|", EngineFactory.Known));
            else engines = ea.Select(x => (string)x!).ToList();
        }

        if (sc["xfail_engines"] != null)
        {
            if (sc["xfail_engines"] is not JsonObject xf) p.Add("xfail_engines: an object, engine -> reason");
            else
            {
                foreach (var (k, v) in xf)
                {
                    if (!EngineFactory.Known.Contains(k)) p.Add($"xfail_engines.{k}: unknown engine");
                    else if (engines.Count > 0 && !engines.Contains(k)) p.Add($"xfail_engines.{k}: the scenario is not run on {k} (engines)");
                    if (v is not JsonValue rv || !rv.TryGetValue<string>(out var why) || why.Length < 8) p.Add($"xfail_engines.{k}: a reason (the open row it waits on)");
                }
            }
        }

        if (sc["setup"] is JsonObject setup) ValidateSetup(setup, p);
        else if (sc["setup"] != null) p.Add("setup: an object");

        var names = new List<string> { "start", "end" };
        if (sc["actions"] != null && sc["actions"] is not JsonArray) p.Add("actions: a list");
        var actions = sc["actions"] as JsonArray ?? new JsonArray();
        for (int i = 0; i < actions.Count; i++)
        {
            string at = $"actions[{i}]";
            if (actions[i] is not JsonObject a) { p.Add(at + ": an object"); continue; }
            CheckKeys(a, at, ActionKeys, ["do"], p);
            string kind = (string?)a["do"] ?? "";
            if (kind is not ("play" or "potion" or "end_turn")) p.Add(at + ".do: play, potion or end_turn");
            if (kind == "play" && (a["card"] == null) == (a["hand_index"] == null)) p.Add(at + ": a play names `card` or `hand_index`, one of them");
            if (kind == "potion" && a["slot"] == null) p.Add(at + ": a potion names `slot`");
            string name = (string?)a["name"] ?? $"a{i + 1}";
            if (names.Contains("after:" + name)) p.Add(at + $".name: '{name}' is used twice");
            names.Add("after:" + name);
            ValidateAnswers(a["answers"], at + ".answers", p);
        }

        ValidateAnswers(sc["start_answers"], "start_answers", p);
        if (sc["expect"] is not JsonArray ex || ex.Count == 0) p.Add("expect: a non-empty list");
        else
        {
            for (int i = 0; i < ex.Count; i++)
            {
                string at = $"expect[{i}]";
                if (ex[i] is not JsonObject e) { p.Add(at + ": an object"); continue; }
                CheckKeys(e, at, ExpectKeys, ["path"], p);
                if (Operators.Count(o => e.ContainsKey(o)) != 1) p.Add(at + ": exactly one of " + string.Join(", ", Operators));
                if (e["at"] != null && !names.Contains((string?)e["at"] ?? "")) p.Add(at + $".at: one of {string.Join(", ", names)}");
                if (e["delta_from"] != null && !names.Contains((string?)e["delta_from"] ?? "")) p.Add(at + $".delta_from: one of {string.Join(", ", names)}");
                if (e["one_of"] != null && e["one_of"] is not JsonArray) p.Add(at + ".one_of: a list");
                try { Resolve(new JsonObject(), (string?)e["path"] ?? ""); }
                catch (FormatException ex2) { p.Add(at + ".path: " + ex2.Message); }
            }
        }

        return p;
    }

    static void CheckKeys(JsonObject o, string where, string[] allowed, string[] required, List<string> p)
    {
        foreach (var kv in o)
        {
            if (!allowed.Contains(kv.Key)) p.Add($"{where}: unknown key '{kv.Key}'");
        }

        foreach (string r in required)
        {
            if (!o.ContainsKey(r)) p.Add($"{where}: missing '{r}'");
        }
    }

    static void ValidateSetup(JsonObject s, List<string> p)
    {
        CheckKeys(s, "setup", SetupKeys, ["character", "ascension", "seed", "encounter", "relics", "deck"], p);
        if ((string?)s["character"] != "IRONCLAD") p.Add("setup.character: IRONCLAD (the only character the engines build)");
        if (s["ascension"] is not JsonValue a || !a.TryGetValue<int>(out var asc) || asc is < 0 or > 10) p.Add("setup.ascension: 0..10");
        if (s["seed"] is not JsonValue sv || !sv.TryGetValue<string>(out var seed) || seed.Length == 0) p.Add("setup.seed: a string");
        if (s["encounter"] is JsonObject enc)
        {
            CheckKeys(enc, "setup.encounter", ["id", "act", "kind", "index"], ["id", "act", "kind", "index"], p);
            if (ActNumber((string?)enc["act"] ?? "") == 0) p.Add("setup.encounter.act: OVERGROWTH, UNDERDOCKS, HIVE or GLORY");
            if ((string?)enc["kind"] is not ("weak" or "normal" or "elite" or "boss")) p.Add("setup.encounter.kind: weak, normal, elite or boss");
        }
        else if (s["encounter"] != null) p.Add("setup.encounter: an object");

        foreach (string k in new[] { "relics", "potions" })
        {
            if (s[k] != null && (s[k] is not JsonArray l || l.Any(x => x is not JsonValue v || !v.TryGetValue<string>(out _)))) p.Add($"setup.{k}: a list of ModelIds");
        }

        foreach (string k in new[] { "deck", "hand", "draw", "discard" })
        {
            if (s[k] == null) continue;
            if (s[k] is not JsonArray l) { p.Add($"setup.{k}: a list of cards"); continue; }
            foreach (var c in l)
            {
                if (c is JsonValue) continue;
                if (c is JsonObject co) CheckKeys(co, $"setup.{k}[]", k is "deck" or "hand" ? ["id", "upgraded"] : ["id", "upgraded", "top"], ["id"], p);
                else p.Add($"setup.{k}: a card is a ModelId or {{id, upgraded}}");
            }
        }

        if (s["hp"] is JsonObject hp) CheckKeys(hp, "setup.hp", ["max", "current"], ["max", "current"], p);
        else if (s["hp"] != null) p.Add("setup.hp: {max, current}");
        if (s["powers"] != null)
        {
            if (s["powers"] is not JsonArray pw) p.Add("setup.powers: a list");
            else
            {
                foreach (var x in pw)
                {
                    if (x is not JsonObject po) { p.Add("setup.powers: objects {id, target, amount}"); continue; }
                    CheckKeys(po, "setup.powers[]", ["id", "target", "amount"], ["id", "target", "amount"], p);
                    string t = (string?)po["target"] ?? "";
                    if (t != "player" && !Regex.IsMatch(t, "^enemy:[0-9]+$")) p.Add($"setup.powers[].target: player or enemy:<slot>, not '{t}'");
                }
            }
        }

        if (s["energy"] != null && (s["energy"] is not JsonValue ev || !ev.TryGetValue<int>(out _))) p.Add("setup.energy: an integer");
    }

    static void ValidateAnswers(JsonNode? answers, string where, List<string> p)
    {
        if (answers == null) return;
        if (answers is not JsonArray list) { p.Add(where + ": a list"); return; }
        foreach (var a in list)
        {
            if (a is not JsonObject o) { p.Add(where + ": objects {pick:[ModelId...]} or {decline:true}"); continue; }
            CheckKeys(o, where + "[]", ["pick", "decline"], [], p);
            if (o.ContainsKey("pick") == o.ContainsKey("decline")) p.Add(where + "[]: `pick` or `decline`, one of them");
            if (o["pick"] != null && o["pick"] is not JsonArray) p.Add(where + "[].pick: a list");
        }
    }

    // ---------------------------------------------------------------------------------------------------- setup to record

    /// <summary>The referee-shaped record (what EmuDump fight and fuzz read) the setup stands for: Setup.cs runs its items.</summary>
    internal static JsonObject BuildRecord(JsonObject setup)
    {
        var enc = setup["encounter"]!.AsObject();
        string act = (string)enc["act"]!;
        var items = new JsonArray();
        if (ActNumber(act) > 1) items.Add(new JsonObject { ["cmd"] = "act", ["index"] = ActNumber(act) });
        items.Add(new JsonObject { ["cmd"] = "relics_set", ["ids"] = Ids(setup["relics"]) });
        items.Add(new JsonObject { ["cmd"] = "deck_set", ["cards"] = new JsonArray(Cards(setup["deck"]).Select(c => (JsonNode)new JsonObject { ["id"] = c.Id, ["upgraded"] = c.Upgraded }).ToArray()) });
        items.Add(new JsonObject { ["cmd"] = "potions_set", ["ids"] = Ids(setup["potions"]) });
        var hp = setup["hp"] as JsonObject;
        items.Add(new JsonObject { ["cmd"] = "hp", ["max"] = (int?)hp?["max"] ?? 80, ["current"] = (int?)hp?["current"] ?? (int?)hp?["max"] ?? 80 });
        items.Add(new JsonObject { ["cmd"] = "fight", ["id"] = (string)enc["id"]! });
        foreach (var pw in setup["powers"] as JsonArray ?? new JsonArray())
        {
            items.Add(new JsonObject { ["cmd"] = "power", ["id"] = (string)pw!["id"]!, ["target"] = (string)pw["target"]!, ["amount"] = (int)pw["amount"]! });
        }

        if (setup["energy"] != null) items.Add(new JsonObject { ["cmd"] = "energy", ["amount"] = (int)setup["energy"]! });
        foreach (var (pile, label) in new[] { ("hand", "Hand"), ("draw", "Draw"), ("discard", "Discard") })
        {
            foreach (var c in Cards(setup[pile]))
            {
                var item = new JsonObject { ["cmd"] = "card", ["id"] = c.Id, ["upgraded"] = c.Upgraded, ["pile"] = label };
                if (c.Top) item["pos"] = "top";
                items.Add(item);
            }
        }

        return new JsonObject
        {
            ["seed"] = (string)setup["seed"]!, ["character"] = (string)setup["character"]!, ["ascension"] = (int)setup["ascension"]!,
            ["neow_pick"] = 0, ["use_potions"] = true, ["setup"] = items,
            ["fight"] = new JsonObject { ["id"] = (string)enc["id"]!, ["act"] = act, ["kind"] = (string)enc["kind"]!, ["index"] = (int)enc["index"]! },
        };
    }

    static JsonArray Ids(JsonNode? n) => new((n as JsonArray ?? new JsonArray()).Select(x => (JsonNode)JsonValue.Create((string)x!)!).ToArray());

    static IEnumerable<(string Id, bool Upgraded, bool Top)> Cards(JsonNode? n) =>
        (n as JsonArray ?? new JsonArray()).Select(c => c is JsonObject o ? ((string)o["id"]!, (bool?)o["upgraded"] == true, (bool?)o["top"] == true) : ((string)c!, false, false));

    // ---------------------------------------------------------------------------------------------------- paths and checks

    /// <summary>
    /// A path into a game view: dot-separated keys, <c>[n]</c> a list index, <c>{ID}</c> the amount of the entry <c>[ID, amount]</c> of a
    /// pair list (powers), <c>#len</c> (last) the length of a list or an object's key count. Null when the path leads nowhere.
    /// Throws FormatException for a malformed path.
    /// </summary>
    internal static JsonNode? Resolve(JsonNode? root, string path)
    {
        if (path.Length == 0) throw new FormatException("empty path");
        var cur = root;
        int i = 0;
        while (i < path.Length)
        {
            char c = path[i];
            if (c == '.')
            {
                i++;
            }
            else if (c == '[' || c == '{')
            {
                int j = path.IndexOf(c == '[' ? ']' : '}', i);
                if (j < 0) throw new FormatException($"unclosed '{c}' in '{path}'");
                string inner = path[(i + 1)..j];
                if (c == '[')
                {
                    if (!int.TryParse(inner, out int n) || n < 0) throw new FormatException($"bad index [{inner}]");
                    cur = cur is JsonArray a && n < a.Count ? a[n] : null;
                }
                else
                {
                    if (inner.Length == 0) throw new FormatException("empty {}");
                    cur = cur is JsonArray pairs ? Entry(pairs, inner) : null;
                }

                i = j + 1;
            }
            else if (c == '#')
            {
                if (path[i..] != "#len") throw new FormatException($"only #len follows '#', in '{path}'");
                return cur switch { JsonArray a => JsonValue.Create(a.Count), JsonObject o => JsonValue.Create(o.Count), _ => null };
            }
            else
            {
                int j = i;
                while (j < path.Length && ".[{#".IndexOf(path[j]) < 0) j++;
                cur = cur is JsonObject o ? o[path[i..j]] : null;
                i = j;
            }
        }

        return cur;
    }

    static JsonNode? Entry(JsonArray pairs, string key)
    {
        foreach (var e in pairs)
        {
            if (e is JsonArray t && t.Count >= 2 && t[0] is JsonValue k && k.TryGetValue<string>(out var s) && s == key) return t.Count == 2 ? t[1] : t;
        }

        return null;
    }

    /// <summary>A number, whatever the node holds (the engines write ints, parsed files hold elements).</summary>
    static bool Num(JsonNode? n, out double d)
    {
        d = 0;
        if (n is not JsonValue v) return false;
        if (v.TryGetValue(out double x)) { d = x; return true; }
        if (v.TryGetValue(out long l)) { d = l; return true; }
        if (v.TryGetValue(out int i)) { d = i; return true; }
        if (v.TryGetValue(out decimal m)) { d = (double)m; return true; }
        if (v.TryGetValue(out float f)) { d = f; return true; }
        return false;
    }

    /// <summary>JSON equality with numbers compared by value (an int and 6.0 are the same).</summary>
    internal static bool Same(JsonNode? a, JsonNode? b)
    {
        if (a == null || b == null) return a == null && b == null;
        if (a is JsonArray x)
        {
            if (b is not JsonArray y || x.Count != y.Count) return false;
            for (int i = 0; i < x.Count; i++)
            {
                if (!Same(x[i], y[i])) return false;
            }

            return true;
        }

        if (a is JsonObject o)
        {
            if (b is not JsonObject q || o.Count != q.Count) return false;
            return o.All(kv => q.ContainsKey(kv.Key) && Same(kv.Value, q[kv.Key]));
        }

        if (Num(a, out double da) && Num(b, out double db)) return da == db;
        return a.ToJsonString() == b.ToJsonString();
    }

    static string Show(JsonNode? n) => n?.ToJsonString() ?? "<absent>";

    /// <summary>The expectations that do not hold at the points the run reached (empty: all hold).</summary>
    internal static List<string> Check(IReadOnlyDictionary<string, JsonObject> points, JsonArray expect)
    {
        var diffs = new List<string>();
        foreach (var item in expect)
        {
            var e = item!.AsObject();
            string at = (string?)e["at"] ?? "end", path = (string)e["path"]!;
            if (!points.TryGetValue(at, out var view)) { diffs.Add($"at {at}: the run never reached this point"); continue; }
            var got = Resolve(view, path);
            string label = $"at {at}, {path}";
            if (e["delta_from"] != null)
            {
                string from = (string)e["delta_from"]!;
                if (!points.TryGetValue(from, out var before)) { diffs.Add($"{label}: the run never reached {from}"); continue; }
                var was = Resolve(before, path);
                if (!Num(got, out double g) || !Num(was, out double w)) { diffs.Add($"{label} (change since {from}): not numeric at both points ({Show(was)} then {Show(got)})"); continue; }
                got = JsonValue.Create(g - w);
                label += $" (change since {from})";
            }

            if (e.ContainsKey("equals"))
            {
                if (!Same(got, e["equals"])) diffs.Add($"{label}: expected {Show(e["equals"])}, got {Show(got)}");
            }
            else if (e.ContainsKey("one_of"))
            {
                if (!e["one_of"]!.AsArray().Any(o => Same(got, o))) diffs.Add($"{label}: expected one of {Show(e["one_of"])}, got {Show(got)}");
            }
            else if (e.ContainsKey("at_least"))
            {
                if (!Num(got, out double v) || v < (double)e["at_least"]!) diffs.Add($"{label}: expected at least {Show(e["at_least"])}, got {Show(got)}");
            }
            else if (e.ContainsKey("at_most"))
            {
                if (!Num(got, out double v) || v > (double)e["at_most"]!) diffs.Add($"{label}: expected at most {Show(e["at_most"])}, got {Show(got)}");
            }
            else if (e.ContainsKey("absent") && (bool)e["absent"]! == (got != null))
            {
                diffs.Add($"{label}: expected {((bool)e["absent"]! ? "absent" : "present")}, got {Show(got)}");
            }
        }

        return diffs;
    }

    // ---------------------------------------------------------------------------------------------------- running

    internal static ScenarioResult Run(string file, JsonNode? node, string engineName, bool keepPoints)
    {
        string name = (node as JsonObject)?["name"]?.ToString() ?? "";
        var problems = Validate(node);
        if (problems.Count > 0) return ScenarioResult.Invalid(file, name, engineName, string.Join("; ", problems));
        var sc = node!.AsObject();
        var res = new ScenarioResult { File = file, Name = name, Engine = engineName, Source = (string)sc["source"]! };
        if (sc["engines"] is JsonArray only && !only.Any(x => (string?)x == engineName))
        {
            res.Status = "skip";
            return res;
        }

        res.Reason = (sc["xfail_engines"] as JsonObject)?[engineName]?.ToString();
        var points = new Dictionary<string, JsonObject>();
        string? error = null;
        try
        {
            Play(sc, engineName, points);
        }
        catch (Exception ex)
        {
            error = $"{ex.GetType().Name}: {FirstLine(ex.Message)}";
        }

        if (error != null) res.Diffs.Add("run stopped: " + error);
        res.Diffs.AddRange(Check(points, sc["expect"]!.AsArray()));
        bool failed = res.Diffs.Count > 0;
        res.Status = !failed ? (res.Reason != null ? "xpass" : "pass") : (res.Reason != null ? "xfail" : "fail");
        if (keepPoints) res.Points = new JsonObject(points.Select(kv => KeyValuePair.Create(kv.Key, (JsonNode?)kv.Value.DeepClone())));
        return res;
    }

    static string FirstLine(string s) => s.Split('\n')[0].Trim();

    static void Play(JsonObject sc, string engineName, Dictionary<string, JsonObject> points)
    {
        var e = EngineFactory.Create(engineName);
        var report = new JsonObject();
        e.StartFight(new FightSpec(BuildRecord(sc["setup"]!.AsObject())), report);
        foreach (var entry in report["setup_log"] as JsonArray ?? new JsonArray())
        {
            if ((bool?)entry!["ok"] == false) throw new InvalidOperationException($"setup item {entry["i"]} ({entry["cmd"]} {entry["id"]}) did not take: {entry["msg"]}");
            if (((string?)entry["msg"] ?? "").StartsWith("refused", StringComparison.Ordinal)) throw new InvalidOperationException($"setup item {entry["i"]} ({entry["cmd"]} {entry["id"]}): {entry["msg"]}");
        }

        if (!e.CombatOpen) throw new InvalidOperationException("the setup left no combat open");
        Answer(e, sc["start_answers"] as JsonArray, "start");
        points["start"] = View(e);
        points["end"] = points["start"];
        var actions = sc["actions"] as JsonArray ?? new JsonArray();
        for (int i = 0; i < actions.Count; i++)
        {
            var a = actions[i]!.AsObject();
            string name = (string?)a["name"] ?? $"a{i + 1}";
            Act(e, a, name);
            Answer(e, a["answers"] as JsonArray, "after " + name);
            points["after:" + name] = View(e);
            points["end"] = points["after:" + name];
        }
    }

    /// <summary>The game view the expectations read: snapshot, net, private, and `screen` (the open screen, or null).</summary>
    static JsonObject View(IEngine e)
    {
        var view = (JsonObject)e.GameView().DeepClone();
        JsonNode? screen = null;
        if (e.Screen is { } s)
        {
            screen = new JsonObject
            {
                ["kind"] = s.Kind, ["max_picks"] = s.MaxPicks, ["skippable"] = s.Skippable, ["source"] = s.SourceModelId,
                ["candidates"] = new JsonArray(s.Candidates.Select(c => (JsonNode)new JsonArray(c.ModelId, c.Upgraded)).ToArray()),
            };
        }

        view["screen"] = screen;
        return view;
    }

    static void Act(IEngine e, JsonObject a, string name)
    {
        if (!e.CombatOpen) throw new InvalidOperationException($"action {name}: the combat is over");
        if (e.Screen is { } open) throw new InvalidOperationException($"action {name}: the {open.Kind} screen is still open");
        string kind = (string)a["do"]!;
        var legal = e.LegalActions();
        string Legal() => string.Join(", ", legal.Select(l => $"{l.Kind}:{l.Index}->[{string.Join(",", l.Targets)}]"));
        if (kind == "end_turn")
        {
            e.EndTurn();
            return;
        }

        int index;
        if (kind == "play")
        {
            index = a["hand_index"] != null ? (int)a["hand_index"]! : HandIndex(e, (string)a["card"]!, (bool?)a["upgraded"]);
        }
        else
        {
            index = (int)a["slot"]!;
        }

        var la = legal.FirstOrDefault(l => l.Kind == kind && l.Index == index)
            ?? throw new InvalidOperationException($"action {name}: {kind} {index} is not legal now (legal: {Legal()})");
        int target;
        if (a["target"] != null) target = (int)a["target"]!;
        else if (la.Targets.Count == 1) target = la.Targets[0];
        else throw new InvalidOperationException($"action {name}: {kind} {index} needs a target (slots {string.Join(",", la.Targets)})");
        if (!la.Targets.Contains(target)) throw new InvalidOperationException($"action {name}: target {target} is not legal for {kind} {index} (legal slots {string.Join(",", la.Targets)})");
        if (kind == "play") e.PlayCard(index, target);
        else e.UsePotion(index, target);
    }

    static int HandIndex(IEngine e, string card, bool? upgraded)
    {
        var hand = e.GameView()["snapshot"]?["hand"] as JsonArray ?? new JsonArray();
        for (int i = 0; i < hand.Count; i++)
        {
            if ((string?)hand[i]![0] == card && (upgraded == null || (bool?)hand[i]![1] == upgraded)) return i;
        }

        throw new InvalidOperationException($"{card} is not in the hand ({string.Join(", ", hand.Select(h => (string?)h![0]))})");
    }

    /// <summary>Answers the open screens in order: a pick takes the first candidate with that ModelId, a decline refuses the screen.</summary>
    static void Answer(IEngine e, JsonArray? answers, string where)
    {
        foreach (var ans in answers ?? new JsonArray())
        {
            var o = ans!.AsObject();
            if (e.Screen is not { } sel) throw new InvalidOperationException($"{where}: an answer, and no screen is open");
            if (o["decline"] != null)
            {
                if (!sel.Skippable) throw new InvalidOperationException($"{where}: the {sel.Kind} screen cannot be declined");
                e.Decline();
                continue;
            }

            foreach (var want in o["pick"]!.AsArray())
            {
                if (e.Screen is not { } now) throw new InvalidOperationException($"{where}: the screen closed before the pick of {want}");
                string id = (string)want!;
                int idx = now.Candidates.ToList().FindIndex(c => c.ModelId == id);
                if (idx < 0)
                    throw new InvalidOperationException($"{where}: {id} is not a candidate of the {now.Kind} screen ({string.Join(", ", now.Candidates.Select(c => c.ModelId))})");
                e.Pick(idx);
            }
        }

        if (e.Screen is { } left)
            throw new InvalidOperationException($"{where}: the {left.Kind} screen is still open ({string.Join(", ", left.Candidates.Select(c => c.ModelId))})");
    }
}
