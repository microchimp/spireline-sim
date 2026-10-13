// Usage: EmuDump <ascension> <seed> [<seed> ...] > out.json   ({"results": [...]} in seed order)
using System.Reflection;
using System.Text.Json;
using System.Text.Json.Nodes;
using Sts2Emulator.Core.Rng;
using Sts2Emulator.Core.Run;

string[] actNames = ["", "OVERGROWTH", "UNDERDOCKS", "HIVE", "GLORY"];
string[] nodeNames = ["Unassigned", "Monster", "Elite", "RestSite", "Shop", "Treasure", "Boss", "Unknown", "Ancient"];
if (args[0] == "pools")
{
    Console.WriteLine(Fight.Pools().ToJsonString());
    return;
}

if (args[0] == "tables")
{
    Console.WriteLine(Tables.Dump().ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
    return;
}

if (args[0] == "fight")
{
    var refs = JsonNode.Parse(File.ReadAllText(args[1]))!["results"]!.AsArray();
    // A fight the game never ran (an error placeholder) has nothing to replay.
    var fights = new JsonArray(refs.Select(r => (JsonNode)(r!["ascension"] == null
        ? new JsonObject { ["seed"] = r["seed"]?.DeepClone(), ["error"] = "no game result to replay", ["steps"] = new JsonArray() }
        : Fight.Replay(r.AsObject()))).ToArray());
    Console.WriteLine(new JsonObject { ["results"] = fights }.ToJsonString(new JsonSerializerOptions { WriteIndented = true }));
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
    // EmuDump reanchor <referee out.json> [--seed SEED]...: each step tried from the game's own state.
    var only = new HashSet<string>();
    HashSet<string>? stale = null;
    for (int i = 2; i < args.Length; i++)
    {
        if (args[i] == "--seed" && i + 1 < args.Length) only.Add(args[i + 1]);
        if (args[i] == "--stale" && i + 1 < args.Length) stale = [.. args[i + 1].Split(',')];
    }

    var mode = new Reanchor.Mode(Scramble: args.Contains("--scramble"), Stale: stale);

    var refs = JsonNode.Parse(File.ReadAllText(args[1]))!["results"]!.AsArray()
        .Where(r => only.Count == 0 || only.Contains((string?)r!["seed"] ?? ""));
    var fights = new JsonArray(refs.Select(r => (JsonNode)(r!["ascension"] == null || r["steps"] == null
        ? new JsonObject { ["seed"] = r["seed"]?.DeepClone(), ["error"] = "no game result to replay", ["steps"] = new JsonArray() }
        : Reanchor.Run(r.AsObject(), mode))).ToArray());
    var doc = Reanchor.PowerKeys();
    doc["results"] = fights;
    Console.WriteLine(doc.ToJsonString());
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
