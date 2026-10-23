// Metamorphic relations over the fights the referee recorded: EmuDump meta <referee out.json>... [options].
// Each distinct start (seed, deck additions, Neow pick, room) is opened in the emulator alone and put through
// the relations of src/Sts2Emulator.Tests/Support/Invariants/Metamorphic.cs. No game is involved: the recording
// only supplies realistic decks and encounters. Prints one JSON document.
using System.Text.Json;
using System.Text.Json.Nodes;
using EmuDump.Engines.Old;
using Sts2Emulator.Tests.Invariants;

static class Meta
{
    /// <summary>FNV-1a: string.GetHashCode differs between processes, and a chooser seed must not.</summary>
    static int StableHash(string s)
    {
        uint h = 2166136261;
        foreach (char c in s)
        {
            h = (h ^ c) * 16777619;
        }

        return (int)(h & 0x3fffffff);
    }

    public static JsonObject Run(string[] args)
    {
        var files = args.Skip(1).Where(a => !a.StartsWith("--") && File.Exists(a)).ToList();
        string? Opt(string name) => args.SkipWhile(a => a != name).Skip(1).FirstOrDefault();
        var only = Opt("--relations")?.Split(',');
        int perFight = int.Parse(Opt("--chooser-seeds") ?? "2");
        int limit = int.Parse(Opt("--limit") ?? "100000");
        int skip = int.Parse(Opt("--skip") ?? "0");
        if (Opt("--diff") is { } dl)
        {
            Metamorphic.DiffLimit = int.Parse(dl);
        }

        int? fixedChooser = Opt("--chooser") is { } ch ? int.Parse(ch) : null;

        var seen = new HashSet<string>();
        var instances = new SortedDictionary<string, int>();
        var failures = new JsonArray();
        int opened = 0, unopenable = 0, fights = 0;
        var errors = new SortedDictionary<string, int>();
        var millis = new SortedDictionary<string, long>();
        foreach (string file in files)
        {
            foreach (var r in JsonNode.Parse(File.ReadAllText(file))!["results"]!.AsArray())
            {
                if (r!["ascension"] == null || r["steps"] == null || r["error"] != null)
                {
                    continue;
                }

                var game = r.AsObject();
                string key = string.Join("|", (string)game["seed"]!, game["ascension"], game["deck_add"]?.ToJsonString(), game["neow_pick"], game["room"]?.ToJsonString());
                if (!seen.Add(key) || seen.Count <= skip || fights >= limit)
                {
                    continue;
                }

                fights++;
                Func<Combat> fresh = () =>
                {
                    var e = OldEngine.EnterRoom(game, new JsonObject());
                    var fork = e.ForkActiveCombat() ?? throw new InvalidOperationException($"no combat after entering the room ({e.State.Phase})");
                    return new Combat(fork.Combat, fork.Rng);
                };
                Combat pristine;
                try
                {
                    pristine = fresh();
                    opened++;
                }
                catch (Exception ex)
                {
                    unopenable++;
                    errors[ex.GetType().Name + ": " + ex.Message.Split('\n')[0]] = errors.GetValueOrDefault(ex.GetType().Name + ": " + ex.Message.Split('\n')[0]) + 1;
                    continue;
                }

                // One opening per fight, copied for each relation; determinism opens from scratch itself.
                Func<Combat> open = () => pristine.Clone();
                int baseSeed = fixedChooser ?? StableHash((string)game["seed"]!);
                for (int k = 0; k < perFight; k++)
                {
                    var report = new RelationReport();
                    Metamorphic.Run(open, baseSeed + k, report, only, k == 0 ? fresh : null);
                    foreach (var (name, n) in report.Instances)
                    {
                        instances[name] = instances.GetValueOrDefault(name) + n;
                    }

                    foreach (var (name, ms) in report.Millis)
                    {
                        millis[name] = millis.GetValueOrDefault(name) + ms;
                    }

                    foreach (var f in report.Failures)
                    {
                        failures.Add(new JsonObject
                        {
                            ["seed"] = (string)game["seed"]!, ["chooser"] = baseSeed + k, ["relation"] = f.Relation,
                            ["step"] = f.Step, ["card"] = f.Card, ["detail"] = f.Detail,
                        });
                    }
                }
            }
        }

        var inst = new JsonObject();
        foreach (var (name, n) in instances)
        {
            inst[name] = n;
        }

        var err = new JsonObject();
        foreach (var (msg, n) in errors)
        {
            err[msg] = n;
        }

        var timing = new JsonObject();
        foreach (var (name, n) in millis)
        {
            timing[name] = n;
        }

        return new JsonObject
        {
            ["ms"] = timing, ["fights"] = fights, ["opened"] = opened, ["unopenable"] = unopenable, ["open_errors"] = err,
            ["instances"] = inst, ["failures"] = failures,
        };
    }
}
