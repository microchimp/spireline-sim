// Run-level modes of EmuDump.
//   EmuDump runstart <ascension> <seed> [<seed> ...]   the run snapshot right after Reset, per seed
//   EmuDump run <referee out.json>                      replays each recorded run decision by decision
using System.Text.Json;
using System.Text.Json.Nodes;
using Sts2Emulator.Core.Run;

static class RunReplay
{
    /// <returns>true when args named a run-level mode (and it ran).</returns>
    public static bool Main(string[] args)
    {
        if (args.Length == 0) return false;
        switch (args[0])
        {
            case "runstart":
            {
                int asc = int.Parse(args[1]);
                var results = new JsonArray();
                foreach (string seed in args.Skip(2))
                {
                    var e = new RunEngine();
                    e.Reset(seed, asc);
                    results.Add(new JsonObject { ["seed"] = e.State.StringSeed, ["snap"] = RunSnap.Export(e) });
                }

                Console.WriteLine(new JsonObject { ["results"] = results }.ToJsonString());
                return true;
            }

            case "run":
            {
                var refs = JsonNode.Parse(File.ReadAllText(args[1]))!["results"]!.AsArray();
                var results = new JsonArray(refs.Select(r => (JsonNode)Replay(r!.AsObject())).ToArray());
                Console.WriteLine(new JsonObject { ["results"] = results }.ToJsonString());
                return true;
            }
        }

        return false;
    }

    static JsonObject Replay(JsonObject game) => new() { ["error"] = "not yet" };
}
