// Hashes of a replay, for mutation testing (tools/fidelity/mutate.py). A mutant's replay is compared
// with the unmutated one step by step; only the fights whose hashes differ are written in full, so a
// run over hundreds of fights writes kilobytes instead of the hundreds of megabytes `fight` writes.
//
//   EmuDump digest fight|reanchor <referee out.json> [--only 3,7,12] [--baseline <digest.jsonl>]
//
// Output, one JSON object per line, flushed as each fight ends (so a hung mutant shows where it hung):
//   {"keys": {...}}            reanchor only, first: Reanchor.PowerKeys(), for reanchor_compare
//   {"i": <index in results>, "f": <hash of the fight minus its steps>, "h": [<hash per step>], "ms": <time>}
//     + "full": <the fight exactly as `fight`/`reanchor` writes it>, when --baseline has other hashes
//     + "cov": [<probe ids>], when the engine was built with mutate.py's probes (Sts2Emulator.MutCov)
using System.Diagnostics;
using System.Reflection;
using System.Security.Cryptography;
using System.Text;
using System.Text.Json.Nodes;

static class Digest
{
    public static void Main(string[] args)
    {
        string mode = args[1];
        var refs = JsonNode.Parse(File.ReadAllText(args[2]))!["results"]!.AsArray();
        var only = new List<int>();
        Dictionary<int, string>? baseline = null;
        for (int k = 3; k + 1 < args.Length; k++)
        {
            if (args[k] == "--only") only.AddRange(args[k + 1].Split(',', StringSplitOptions.RemoveEmptyEntries).Select(int.Parse));
            if (args[k] == "--baseline") baseline = LoadBaseline(args[k + 1]);
        }

        if (only.Count == 0) only.AddRange(Enumerable.Range(0, refs.Count));
        // Probes exist only in mutate.py's instrumented scratch build; the normal engine has none.
        var take = typeof(Sts2Emulator.Core.CombatState).Assembly.GetType("Sts2Emulator.MutCov")?.GetMethod("Take", BindingFlags.Public | BindingFlags.Static);
        var stdout = new StreamWriter(Console.OpenStandardOutput(), new UTF8Encoding(false)) { AutoFlush = true };
        if (mode == "reanchor") stdout.WriteLine(new JsonObject { ["keys"] = Reanchor.PowerKeys() }.ToJsonString());
        foreach (int i in only)
        {
            var game = refs[i]!.AsObject();
            var sw = Stopwatch.StartNew();
            take?.Invoke(null, null);  // whatever ran before this fight (static setup) is not its own
            JsonObject result = game["ascension"] == null || game["steps"] == null
                ? new JsonObject { ["seed"] = game["seed"]?.DeepClone(), ["error"] = "no game result to replay", ["steps"] = new JsonArray() }
                : mode == "reanchor" ? Reanchor.Run(game) : Fight.Replay(game);
            var line = Hashes(result);
            line["i"] = i;
            line["ms"] = sw.ElapsedMilliseconds;
            if (take != null) line["cov"] = new JsonArray(((int[])take.Invoke(null, null)!).Select(p => (JsonNode)p).ToArray());
            if (baseline != null && (!baseline.TryGetValue(i, out var want) || want != Key(line)))
            {
                line["full"] = result;
            }

            stdout.WriteLine(line.ToJsonString());
        }
    }

    /// <summary>"f" and "h" of one fight: its own fields, then each step, hashed separately.</summary>
    static JsonObject Hashes(JsonObject fight)
    {
        var steps = fight["steps"]?.AsArray() ?? new JsonArray();
        var rest = new JsonObject();
        foreach (var (k, v) in fight)
        {
            if (k != "steps") rest[k] = v?.DeepClone();
        }

        return new JsonObject
        {
            ["f"] = Hash(rest.ToJsonString()),
            ["h"] = new JsonArray(steps.Select(s => (JsonNode)Hash(s?.ToJsonString() ?? "null")).ToArray()),
        };
    }

    static string Hash(string s) => Convert.ToHexString(SHA1.HashData(Encoding.UTF8.GetBytes(s)))[..16];

    static string Key(JsonObject line) => (string)line["f"]! + "|" + string.Join(",", line["h"]!.AsArray().Select(x => (string)x!));

    static Dictionary<int, string> LoadBaseline(string path)
    {
        var d = new Dictionary<int, string>();
        foreach (var raw in File.ReadLines(path))
        {
            var o = JsonNode.Parse(raw)!.AsObject();
            if (o["i"] != null) d[(int)o["i"]!] = Key(o);
        }

        return d;
    }
}
