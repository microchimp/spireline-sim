// `EmuDump codex <runs.jsonl[.gz]> <out.jsonl> [options]`: the codex replay over a file of run
// histories (one JSON object a line, the game's run-history schema 9 as Spire Codex serves it).
// Reads the input as data only. Writes one result line per run walked, and a summary to stderr.
//
//   --decompiled DIR   the decompiled game's root (event option slots are read off its events)
//   --character NAME   IRONCLAD (default) or any; runs of other characters are skipped
//   --ascension N      only this ascension (default: any the engine models)
//   --limit N          stop after N runs walked
//   --shard I/N        walk only the runs whose index mod N is I
using System.Diagnostics;
using System.IO.Compression;
using System.Text.Json;
using System.Text.Json.Nodes;
using Sts2Emulator.Core;

namespace EmuDump.Codex;

public static class Driver
{
    public static int Run(string[] args)
    {
        if (args.Length < 2)
        {
            Console.Error.WriteLine("usage: EmuDump codex <runs.jsonl[.gz]> <out.jsonl> [--decompiled DIR] [--character NAME|any] [--ascension N] [--limit N] [--shard I/N]");
            return 2;
        }

        string input = args[0], output = args[1];
        string? decompiled = null;
        string character = "IRONCLAD";
        int? ascension = null;
        int limit = int.MaxValue, shard = 0, shards = 1;
        for (int i = 2; i + 1 < args.Length; i += 2)
        {
            switch (args[i])
            {
                case "--decompiled": decompiled = args[i + 1]; break;
                case "--character": character = args[i + 1].ToUpperInvariant(); break;
                case "--ascension": ascension = int.Parse(args[i + 1]); break;
                case "--limit": limit = int.Parse(args[i + 1]); break;
                case "--shard":
                    var p = args[i + 1].Split('/');
                    shard = int.Parse(p[0]);
                    shards = int.Parse(p[1]);
                    break;
                default:
                    Console.Error.WriteLine($"unknown option {args[i]}");
                    return 2;
            }
        }

        string? eventsDir = decompiled is null ? null : Path.Combine(decompiled, "MegaCrit.Sts2.Core.Models.Events");
        string? epochsDir = decompiled is null ? null : Path.Combine(decompiled, "MegaCrit.Sts2.Core.Timeline.Epochs");
        var walker = new Walker(new EventOptionIndex(eventsDir), new ProfileIndex(epochsDir));
        using var inStream = OpenInput(input);
        using var reader = new StreamReader(inStream);
        using var writer = new StreamWriter(output);
        var skipped = new Dictionary<string, int>();
        int index = -1, walked = 0;
        var clock = Stopwatch.StartNew();
        double walkSeconds = 0;
        string? line;
        while ((line = reader.ReadLine()) is not null && walked < limit)
        {
            index++;
            if (index % shards != shard || line.Length == 0)
            {
                continue;
            }

            using var doc = JsonDocument.Parse(line);
            var rec = doc.RootElement;
            // A mod's own saved data (BaseLib's card modifiers, a mod's prefixed card props) says
            // the game that played the run was modded, whatever its ids look like.
            string? why = line.Contains("BaseLib", StringComparison.Ordinal) || line.Contains("save_dict_", StringComparison.Ordinal)
                    || line.Contains("\"CardModel_", StringComparison.Ordinal)
                ? "mod data in the record (modded)"
                : ModdedId.IsMatch(line)
                    ? "modded content id (modded)"
                    : Skip(rec, character, ascension);
            if (why is not null)
            {
                skipped[why] = skipped.GetValueOrDefault(why) + 1;
                continue;
            }

            long t0 = Stopwatch.GetTimestamp();
            JsonObject result = walker.Walk(rec);
            double sec = Stopwatch.GetElapsedTime(t0).TotalSeconds;
            walkSeconds += sec;
            result["character"] = Names.Bare(rec.GetProperty("players")[0].GetProperty("character").GetString());
            result["ms"] = Math.Round(sec * 1000, 3);
            writer.WriteLine(result.ToJsonString());
            walked++;
        }

        var summary = new JsonObject
        {
            ["walked"] = walked,
            ["skipped"] = new JsonObject(skipped.Select(kv => KeyValuePair.Create(kv.Key, (JsonNode?)kv.Value))),
            ["seconds_total"] = Math.Round(clock.Elapsed.TotalSeconds, 2),
            ["seconds_walking"] = Math.Round(walkSeconds, 2),
            ["runs_per_second_walking"] = walkSeconds > 0 ? Math.Round(walked / walkSeconds, 1) : 0,
            ["seconds_in_reset"] = Math.Round(Walker.ResetSeconds, 2),
            ["seconds_in_fight_resolve"] = Math.Round(Walker.FightSeconds, 2),
        };
        Console.Error.WriteLine(summary.ToJsonString());
        return 0;
    }

    static Stream OpenInput(string path)
    {
        Stream s = File.OpenRead(path);
        return path.EndsWith(".gz", StringComparison.OrdinalIgnoreCase) ? new GZipStream(s, CompressionMode.Decompress) : s;
    }

    /// <summary>
    /// A mod's own content id: the game's ids are `KIND.UPPER_SNAKE`, and mods prefix theirs
    /// with the mod's name and a hyphen (`ACT.ACTSFROMTHEPAST-...`, `EVENT.ACTSFROMTHEPAST-...`).
    /// </summary>
    public static readonly System.Text.RegularExpressions.Regex ModdedId = new(
        "\"(ACT|EVENT|ENCOUNTER|MONSTER|CARD|RELIC|POTION|ENCHANTMENT|MODIFIER|CHARACTER)\\.[A-Z0-9_]*-",
        System.Text.RegularExpressions.RegexOptions.Compiled);

    /// <summary>The acts the game has; any other act id is a mod's.</summary>
    static readonly HashSet<string> VanillaActs = ["OVERGROWTH", "UNDERDOCKS", "HIVE", "GLORY"];

    /// <summary>Why a run is not walked, or null.</summary>
    public static string? Skip(JsonElement rec, string character, int? ascension)
    {
        if (rec.TryGetProperty("acts", out var acts) && acts.ValueKind == JsonValueKind.Array
            && acts.EnumerateArray().Any(a => !VanillaActs.Contains(Names.Bare(a.GetString()))))
        {
            return "modded act (modded)";
        }

        if (!rec.TryGetProperty("players", out var players) || players.GetArrayLength() != 1)
        {
            return "not single-player";
        }

        if (rec.GetProperty("game_mode").GetString() != "standard")
        {
            return "not standard mode";
        }

        if (rec.TryGetProperty("modifiers", out var mods) && mods.ValueKind == JsonValueKind.Array && mods.GetArrayLength() > 0)
        {
            return "modifiers";
        }

        string ch = Names.Bare(players[0].GetProperty("character").GetString());
        if (character != "ANY" && ch != character)
        {
            return "other character";
        }

        int asc = rec.GetProperty("ascension").GetInt32();
        if (asc < 0 || asc > Ascension.MaxRunLevel)
        {
            return "ascension outside the engine";
        }

        if (ascension is { } want && asc != want)
        {
            return "other ascension";
        }

        if (Names.Character(ch) is null)
        {
            return "character the emulator does not have";
        }

        // Content the emulator does not know is a modded run.
        foreach (var c in players[0].GetProperty("deck").EnumerateArray())
        {
            if (!Names.CardIds.ContainsKey(Names.Bare(c.GetProperty("id").GetString())))
            {
                return "unknown card (modded)";
            }
        }

        foreach (var r in players[0].GetProperty("relics").EnumerateArray())
        {
            if (!Names.RelicIds.ContainsKey(Names.Bare(r.GetProperty("id").GetString())))
            {
                return "unknown relic (modded)";
            }
        }

        return null;
    }
}
