// Live-state invariants over the fights the referee recorded: EmuDump invariants <referee out.json>... [--dir <jobs dir>].
// Each distinct clean fight is replayed exactly as `EmuDump fight` replays it (Fight.Replay), and after the opening
// and after every step the engine's own state is judged by StateInvariants: what no snapshot shows (queues and held
// cards at rest, generators shared between fields, the known-order counters), plus the streams and counters that
// must never fall. Prints one JSON document. The recording only supplies the actions; no game state is compared.
using System.Security.Cryptography;
using System.Text;
using System.Text.Json;
using System.Text.Json.Nodes;
using Sts2Emulator.Core.Run;
using Sts2Emulator.Tests.Invariants;

static class Live
{
    public static JsonObject Run(string[] args)
    {
        var files = args.Skip(1).Where(a => !a.StartsWith("--") && File.Exists(a)).ToList();
        int dir = Array.IndexOf(args, "--dir");
        if (dir >= 0 && dir + 1 < args.Length)
        {
            files.AddRange(Directory.GetFiles(args[dir + 1], "*.out.json").OrderBy(f => f, StringComparer.Ordinal));
        }

        var seen = new HashSet<string>();
        int fights = 0, steps = 0, replayErrors = 0;
        var byCheck = new SortedDictionary<string, int>();
        var fightsWith = new SortedDictionary<string, int>();
        var examples = new JsonArray();
        foreach (string file in files)
        {
            foreach (var r in JsonNode.Parse(File.ReadAllText(file))!["results"]!.AsArray())
            {
                if (r!["ascension"] == null || r["steps"] == null || r["error"] != null || r["steps"]!.AsArray().Count == 0)
                {
                    continue;
                }

                var game = r.AsObject();
                string key = Convert.ToHexString(SHA1.HashData(Encoding.UTF8.GetBytes(
                    game["seed"] + "|" + game["ascension"] + "|" + game["deck_add"]?.ToJsonString() + "|" + game["neow_pick"] + "|"
                    + game["room"]?.ToJsonString() + "|" + string.Join(",", game["steps"]!.AsArray().Select(s => s!["action"]?.ToJsonString())))));
                if (!seen.Add(key))
                {
                    continue;
                }

                fights++;
                var tracker = new StateInvariants.Tracker();
                var broken = new HashSet<string>();
                object? current = null;
                Fight.AfterStep = (engine, step) =>
                {
                    if (engine.State.Phase != RunPhase.Combat || engine.State.ActiveCombat is not { } combat)
                    {
                        return;
                    }

                    // A recorded action that outlives the fight (the driver kept playing) can walk the replay into the
                    // next room: a different combat, so the counters start again.
                    if (!ReferenceEquals(current, combat))
                    {
                        current = combat;
                        tracker = new StateInvariants.Tracker();
                    }

                    steps++;
                    foreach (var v in tracker.Step(combat))
                    {
                        byCheck[v.Check] = byCheck.GetValueOrDefault(v.Check) + 1;
                        if (broken.Add(v.Check))
                        {
                            fightsWith[v.Check] = fightsWith.GetValueOrDefault(v.Check) + 1;
                        }

                        if (examples.Count < 40)
                        {
                            examples.Add(new JsonObject { ["seed"] = (string)game["seed"]!, ["step"] = step, ["check"] = v.Check, ["detail"] = v.Detail });
                        }
                    }
                };
                try
                {
                    var replayed = Fight.Replay(game);
                    if (replayed["error"] != null)
                    {
                        replayErrors++;
                    }
                }
                finally
                {
                    Fight.AfterStep = null;
                }
            }
        }

        var checks = new JsonObject();
        foreach (var (name, n) in byCheck)
        {
            checks[name] = new JsonObject { ["violations"] = n, ["fights"] = fightsWith[name] };
        }

        return new JsonObject { ["fights"] = fights, ["steps"] = steps, ["replay_errors"] = replayErrors, ["checks"] = checks, ["examples"] = examples };
    }
}
