// Usage: EmuDump <ascension> <seed> [<seed> ...] > out.json   ({"results": [...]} in seed order)
//        EmuDump codex <runs.jsonl[.gz]> <out.jsonl> [...]   (see Codex/Driver.cs)
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Nodes;
using Sts2Emulator.Core.Rng;
using Sts2Emulator.Core.Run;

if (!System.Runtime.CompilerServices.RuntimeFeature.IsDynamicCodeSupported && !AotChecked.Allows(args))
{
    // The NativeAOT build (aot.sh) is checked byte for byte against this one on the corpus for
    // these calls only; the other modes read the engine through reflection, which it trims.
    Console.Error.WriteLine("EmuDump (NativeAOT): only `fight <out.json> [--compact]` and `reanchor <out.json> [--seed S]...`; "
        + "run everything else with `dotnet EmuDump.dll`");
    Environment.Exit(2);
}

string[] actNames = ["", "OVERGROWTH", "UNDERDOCKS", "HIVE", "GLORY"];
string[] nodeNames = ["Unassigned", "Monster", "Elite", "RestSite", "Shop", "Treasure", "Boss", "Unknown", "Ancient"];
if (RunReplay.Main(args))  // run-level modes (RunReplay.cs): runstart, run
{
    return;
}

if (args[0] == "codex")
{
    // EmuDump codex <runs.jsonl[.gz]> <out.jsonl> [...]: real run histories against the run engine.
    Environment.Exit(EmuDump.Codex.Driver.Run(args[1..]));
}

if (args[0] == "pools")
{
    Console.WriteLine(Fight.Pools().ToJsonString());
    return;
}

if (args[0] == "setupinfo")
{
    // What the setup mirror can build, for referee/scenarios.py: each relic's pickup follow-up.
    Console.WriteLine(Setup.Info().ToJsonString());
    return;
}

if (args[0] == "tables")
{
    Console.WriteLine(Tables.Dump().ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
    return;
}

if (args[0] == "fight")
{
    Prof.Start();
    if (args.Contains("--netstate"))
    {
        // The game's NetFullCombatState fields after each step (NetState.cs), for netstate_diff.
        Fight.OnStep += (e, step) => step["net"] = new JsonArray(new JsonObject { ["ctx"] = "step", ["state"] = NetState.From(e) });
    }

    if (args.Any(a => a.StartsWith("--state-hash", StringComparison.Ordinal)))
    {
        // The full-state hash after each step (StateWalk.cs: every RunState field, the combat's
        // included, not the dump's bytes), for tools/fidelity/corpus.py identity. --state-hash=fields
        // adds one short hash per field, so a difference names the field.
        bool fields = args.Contains("--state-hash=fields");
        Fight.OnStep += (e, step) =>
        {
            step["state_hash"] = StateWalk.RunHash(e);
            if (!fields) return;
            var o = new JsonObject();
            foreach (var (k, v) in StateWalk.FieldHashes(e.State)) o[k] = v;
            if (e.State.ActiveCombat is { } c)
            {
                foreach (var (k, v) in StateWalk.FieldHashes(c)) o["ActiveCombat." + k] = v;
            }

            step["state_fields"] = o;
        };
    }

    if (args.Contains("--rng-log"))
    {
        // Every draw of a combat stream during each step, [stream, index, caller] (RngLog.cs), for
        // tools/fidelity/rng_diff.py.
        RngLog.Install();
        Fight.OnStep += (e, step) => step["rng_log"] = RngLog.Take(e);
    }

    JsonArray refs;
    using (Prof.Scope("read_input")) refs = JsonNode.Parse(File.ReadAllText(args[1]))!["results"]!.AsArray();
    // A fight the game never ran (an error placeholder) has nothing to replay.
    var fights = new JsonArray(refs.Select(r =>
    {
        RngLog.Clear();  // what a fight that threw before its first step drew is not the next one's
        return (JsonNode)(r!["ascension"] == null
            ? new JsonObject { ["seed"] = r["seed"]?.DeepClone(), ["error"] = "no game result to replay", ["steps"] = new JsonArray() }
            : Fight.Replay(r.AsObject()));
    }).ToArray());
    string text;
    using (Prof.Scope("serialize")) text = new JsonObject { ["results"] = fights }.ToJsonString(new JsonSerializerOptions { WriteIndented = !args.Contains("--compact") });
    using (Prof.Scope("write")) StdOut.WriteLine(text);
    Prof.Report("fight", fights.Count);
    return;
}

if (args[0] == "invariants")
{
    Console.WriteLine(Live.Run(args).ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
    return;
}

if (args[0] == "meta")
{
    Console.WriteLine(Meta.Run(args).ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
    return;
}

if (args[0] == "muthost")
{
    Environment.Exit(MutHost.Main(args));  // many mutants in one process (MutHost.cs)
}

if (args[0] == "digest")
{
    Digest.Main(args);  // hashes per step, for mutation testing (Digest.cs)
    return;
}

if (args[0] == "reanchor")
{
    // EmuDump reanchor <referee out.json> [--seed SEED]... [--scramble] [--stale ...] [--perturb-moves] [--trace] [--trace-to STEP]:
    // each step tried from the game's own state; --trace records what happens inside each step (Trace.cs).
    var only = new HashSet<string>();
    HashSet<string>? stale = null;
    for (int i = 2; i < args.Length; i++)
    {
        if (args[i] == "--seed" && i + 1 < args.Length) only.Add(args[i + 1]);
        if (args[i] == "--stale" && i + 1 < args.Length) stale = [.. args[i + 1].Split(',')];
    }

    int? traceTo = null;
    for (int i = 2; i < args.Length - 1; i++)
    {
        if (args[i] == "--trace-to") traceTo = int.Parse(args[i + 1]);
    }

    Prof.Start();
    var mode = new Reanchor.Mode(Scramble: args.Contains("--scramble"), Stale: stale, PerturbMoves: args.Contains("--perturb-moves"),
        Trace: args.Contains("--trace") || traceTo != null, TraceTo: traceTo);

    JsonArray all;
    using (Prof.Scope("read_input")) all = JsonNode.Parse(File.ReadAllText(args[1]))!["results"]!.AsArray();
    var refs = all
        .Where(r => only.Count == 0 || only.Contains((string?)r!["seed"] ?? ""));
    var fights = new JsonArray(refs.Select(r => (JsonNode)(r!["ascension"] == null || r["steps"] == null
        ? new JsonObject { ["seed"] = r["seed"]?.DeepClone(), ["error"] = "no game result to replay", ["steps"] = new JsonArray() }
        : Reanchor.Run(r.AsObject(), mode))).ToArray());
    var doc = Reanchor.PowerKeys();
    doc["results"] = fights;
    string text;
    using (Prof.Scope("serialize")) text = doc.ToJsonString();
    using (Prof.Scope("write")) StdOut.WriteLine(text);
    Prof.Report("reanchor", fights.Count);
    return;
}

int ascension = int.Parse(args[0]);
var results = new List<object>();
foreach (string seed in args.Skip(1))
{
    var engine = new RunEngine();
    engine.Reset(seed, ascension);
    results.Add(Dump(engine.State, ascension));
}

Console.WriteLine(JsonSerializer.Serialize(new { results }, new JsonSerializerOptions { WriteIndented = true }));

static Dictionary<string, object> Counters(object set) =>
    set.GetType().GetProperties()
        .Where(p => p.PropertyType == typeof(GameRng))
        .ToDictionary(p => p.Name, p => (object)new { counter = ((GameRng)p.GetValue(set)!).CallCount });

object Dump(RunState s, int ascension) => new
{
    seed = s.StringSeed,
    ascension,
    acts = s.Acts.Select(a => actNames[a.Act]).ToArray(),
    rng = Counters(s.Rng),
    player_rng = Counters(s.PlayerRng),
    map = s.MapNodes.Values.OrderBy(n => n.Row).ThenBy(n => n.Col).Select(n => new
    {
        col = n.Col,
        row = n.Row,
        type = nodeNames[n.NodeType],
        children = n.Children.OrderBy(c => c.Row).ThenBy(c => c.Col).Select(c => new[] { c.Col, c.Row }).ToArray(),
    }).ToArray(),
};

/// <summary>The calls the NativeAOT build is verified for (see tools/emudump/aot.sh).</summary>
static class AotChecked
{
    internal static bool Allows(string[] args)
    {
        if (args.Length < 2) return false;
        if (args[0] == "fight") return args.Skip(2).All(a => a == "--compact");
        if (args[0] != "reanchor") return false;
        for (int i = 2; i < args.Length; i += 2)
        {
            if (args[i] != "--seed" || i + 1 >= args.Length) return false;
        }

        return true;
    }
}
