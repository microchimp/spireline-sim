// Many mutants in one process, for tools/fidelity/mutate.py's schemata mode. The engine it runs is a
// scratch build in which every sampled mutant is compiled in behind `MutCov.Is(id)` and every block
// carries a coverage probe (Sts2Emulator.MutCov, written by mutate.py, never committed). Per fight:
// the run is set up once with no mutant active and remembered (RunEngine.Reset is the costly part of
// a replay and no mutant is in it), the unmutated replay is checked against the baseline hashes, and
// then each mutant whose block that fight ran is replayed (plain and re-anchored) from the remembered
// setup. Only mutants whose hashes differ are written in full.
//
//   EmuDump muthost <referee out.json> --fights i:pos,... --mutants id:unit,... --baseline <prefix> [--kills <file>]
//
// Output lines: {"base": i, "ok": bool}  {"b": [i, id]} before each mutant run (a hang shows where)
//   {"i": i, "m": id, "fight": <full or null>, "reanchor": <full or null>} when a hash differs
//   {"done": i, "ran": [ids]}. Exit 3 when the unmutated replay no longer matches (a mutant left static
//   state behind): the caller restarts from that fight. --kills lists "id pos" lines the caller writes as
//   it judges; a mutant killed at or before a fight's position is not run on it.
using System.Reflection;
using System.Text;
using System.Text.Json.Nodes;
using Sts2Emulator.Core.Run;

static class MutHost
{
    public static int Main(string[] args)
    {
        var refs = JsonNode.Parse(File.ReadAllText(args[1]))!["results"]!.AsArray();
        var fights = new List<(int I, int Pos)>();
        var mutants = new List<(int Id, int Unit)>();
        string? baselinePrefix = null, killsPath = null;
        for (int k = 2; k + 1 < args.Length; k++)
        {
            if (args[k] == "--fights") fights.AddRange(Pairs(args[k + 1]));
            if (args[k] == "--mutants") mutants.AddRange(Pairs(args[k + 1]));
            if (args[k] == "--baseline") baselinePrefix = args[k + 1];
            if (args[k] == "--kills") killsPath = args[k + 1];
        }

        var cov = typeof(Sts2Emulator.Core.CombatState).Assembly.GetType("Sts2Emulator.MutCov")
            ?? throw new InvalidOperationException("muthost needs an engine built by tools/fidelity/mutate.py");
        var take = cov.GetMethod("Take")!;
        var forget = cov.GetMethod("Forget")!;
        var active = cov.GetField("Active")!;
        var noMemo = cov.GetField("NoMemo")!;
        var baseFight = Baseline(baselinePrefix + ".fight.jsonl");
        var baseRe = Baseline(baselinePrefix + ".reanchor.jsonl");
        var killedAt = new Dictionary<int, int>();
        DateTime killsSeen = DateTime.MinValue;
        var stdout = new StreamWriter(Console.OpenStandardOutput(), new UTF8Encoding(false)) { AutoFlush = true };
        foreach (var (i, pos) in fights)
        {
            var game = refs[i]!.AsObject();
            active.SetValue(null, -1);
            noMemo.SetValue(null, false);
            forget.Invoke(null, null);
            take.Invoke(null, null);
            new RunEngine().Reset((string)game["seed"]!, (int)game["ascension"]!);  // remembered for this fight
            var resetUnits = ((int[])take.Invoke(null, null)!).ToHashSet();
            var f0 = Fight.Replay(game);
            var r0 = Reanchor.Run(game);
            var ran = ((int[])take.Invoke(null, null)!).ToHashSet();
            ran.UnionWith(resetUnits);
            bool ok = Key(f0) == baseFight.GetValueOrDefault(i) && Key(r0) == baseRe.GetValueOrDefault(i);
            stdout.WriteLine(new JsonObject { ["base"] = i, ["ok"] = ok }.ToJsonString());
            if (!ok) return 3;
            if (killsPath != null && File.Exists(killsPath) && File.GetLastWriteTimeUtc(killsPath) != killsSeen)
            {
                try
                {
                    // The caller appends while we read: share the file, and try again next fight if it is busy.
                    var stamp = File.GetLastWriteTimeUtc(killsPath);
                    using var reader = new StreamReader(new FileStream(killsPath, FileMode.Open, FileAccess.Read, FileShare.ReadWrite | FileShare.Delete));
                    while (reader.ReadLine() is { } line)
                    {
                        var p = line.Split(' ');
                        if (p.Length == 2 && int.TryParse(p[0], out int id) && int.TryParse(p[1], out int at))
                            killedAt[id] = Math.Min(at, killedAt.GetValueOrDefault(id, int.MaxValue));
                    }

                    killsSeen = stamp;
                }
                catch (IOException)
                {
                }
            }

            var done = new JsonArray();
            foreach (var (id, unit) in mutants)
            {
                if (!ran.Contains(unit) || killedAt.GetValueOrDefault(id, int.MaxValue) <= pos) continue;
                stdout.WriteLine(new JsonObject { ["b"] = new JsonArray(i, id) }.ToJsonString());
                active.SetValue(null, id);
                noMemo.SetValue(null, resetUnits.Contains(unit));  // its block runs in the setup: no shortcut
                var f = Guard(() => Fight.Replay(game));
                var r = Guard(() => Reanchor.Run(game));
                active.SetValue(null, -1);
                noMemo.SetValue(null, false);
                bool fd = Key(f) != baseFight.GetValueOrDefault(i), rd = Key(r) != baseRe.GetValueOrDefault(i);
                if (fd || rd)
                {
                    stdout.WriteLine(new JsonObject { ["i"] = i, ["m"] = id, ["fight"] = fd ? f : null, ["reanchor"] = rd ? r : null }.ToJsonString());
                }

                done.Add(id);
            }

            take.Invoke(null, null);
            stdout.WriteLine(new JsonObject { ["done"] = i, ["ran"] = done }.ToJsonString());
        }

        return 0;
    }

    static IEnumerable<(int, int)> Pairs(string s) => s.Split(',', StringSplitOptions.RemoveEmptyEntries)
        .Select(p => p.Split(':')).Select(p => (int.Parse(p[0]), int.Parse(p[1])));

    /// <summary>A replay that threw outside its own handler (the mutant broke the setup) is an error result.</summary>
    static JsonObject Guard(Func<JsonObject> run)
    {
        try
        {
            return run();
        }
        catch (Exception ex)
        {
            return new JsonObject { ["error"] = ex.GetType().Name + ": " + ex.Message, ["steps"] = new JsonArray() };
        }
    }

    /// <summary>The same key as Digest's: the fight's own fields and each step, hashed.</summary>
    static string Key(JsonObject fight) => Digest.KeyOf(fight);

    static Dictionary<int, string> Baseline(string path) => Digest.LoadBaseline(path);
}
