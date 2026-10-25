// EmuDump fuzz <specs.json> [--fields] [--engine old|core2]: seeded setup fights (tools/fidelity/fuzz_core2.py writes the
// specs), each played to the end by a random legal player and written in the shape `EmuDump fight --state-hash` reads:
// the actions taken, the screen answers (as the referee's `choices`, tagged with their step) and the state hash after
// each step. Both engines are given the same spec: where their states agree the player makes the same moves, so the
// first step whose action or hash differs is the first difference. A spec's record plus the output's `room`, `choices`
// and the actions as `steps` is a referee-shaped record: `EmuDump fight` replays it step for step on either engine.
// The engine is reached through IEngine only.
using System.Text.Json.Nodes;
using EmuDump.Engine;
using EmuDump.Engines;

static class Fuzz
{
    internal static int Main(string[] args)
    {
        if (args.Length < 1)
        {
            Console.Error.WriteLine("usage: EmuDump fuzz <specs.json> [--fields] [--engine old|core2]");
            return 2;
        }

        bool fields = args.Contains("--fields");
        var specs = JsonNode.Parse(File.ReadAllText(args[0]))!["results"]!.AsArray();
        var results = new JsonArray(specs.Select(s => (JsonNode)Play(s!.AsObject(), fields)).ToArray());
        StdOut.WriteLine(new JsonObject { ["results"] = results }.ToJsonString());
        return 0;
    }

    /// <summary>One spec's fight. The spec's <c>fuzz</c> block: seed (the player's), max_steps (80), p_end (0.1), p_decline (0.25), roster (write the opening enemies as `roster`). <paramref name="make"/> makes the engine (default: the process's <c>--engine</c>); tests use it to pick the core2 switches explicitly.</summary>
    internal static JsonObject Play(JsonObject rec, bool fields, Func<IEngine>? make = null)
    {
        var fz = rec["fuzz"] as JsonObject ?? new JsonObject();
        var rng = new Mix((ulong)((long?)fz["seed"] ?? 1));
        int maxSteps = (int?)fz["max_steps"] ?? 80;
        double pEnd = (double?)fz["p_end"] ?? 0.1;
        double pDecline = (double?)fz["p_decline"] ?? 0.25;
        var outp = new JsonObject();
        var steps = new JsonArray();
        var choices = new JsonArray();
        outp["steps"] = steps;
        outp["choices"] = choices;
        try
        {
            var e = (make ?? (() => EngineFactory.Create()))();
            e.StartFight(new FightSpec(rec), outp);
            if ((bool?)fz["roster"] == true)
            {
                // fuzz_core2.py's catalog: the enemies the encounter opens with, as the game names them.
                outp["roster"] = new JsonArray((e.GameView()["snapshot"]?["enemies"] as JsonArray ?? new JsonArray())
                    .Select(x => (JsonNode)JsonValue.Create((string?)x!["name"])!).ToArray());
            }

            Answer(e, 0, rng, pDecline, choices);
            steps.Add(Step(e, null, fields));
            Agree(e, 0, outp);
            for (int n = 1; n <= maxSteps && e.CombatOpen; n++)
            {
                var legal = e.LegalActions();
                if (legal.Count == 0) break;
                var rest = legal.Where(a => a.Kind != "end_turn").ToList();
                var pick = rest.Count == 0 || rng.Chance(pEnd) ? legal.First(a => a.Kind == "end_turn") : rest[rng.Next(rest.Count)];
                int target = pick.Targets[rng.Next(pick.Targets.Count)];
                var action = new JsonObject { ["kind"] = pick.Kind };
                switch (pick.Kind)
                {
                    case "play":
                        action["hand_index"] = pick.Index;
                        action["target_slot"] = target;
                        e.PlayCard(pick.Index, target);
                        break;
                    case "potion":
                        action["slot"] = pick.Index;
                        action["target_slot"] = target;
                        e.UsePotion(pick.Index, target);
                        break;
                    default:
                        e.EndTurn();
                        break;
                }

                Answer(e, n, rng, pDecline, choices);
                steps.Add(Step(e, action, fields));
                Agree(e, n, outp);
            }
        }
        catch (Exception ex)
        {
            outp["error"] = ex.ToString();
        }

        return outp;
    }

    /// <summary>
    /// The agreement report (F10 on): the old ending predicate is compared with the kernel's after every step. Nothing here
    /// fails on a difference; <c>tools/fidelity/fuzz_core2.py</c> fails on one outside its EXPECTED_ENDING_DISAGREE. A step
    /// where they differ is listed under the output's <c>ending_disagree</c> (step and the two readings); the key is absent when
    /// they agree throughout, so the old engine's output and an agreeing core2 run are unchanged.
    /// </summary>
    static void Agree(IEngine e, int step, JsonObject outp)
    {
        if (e.EndingDisagreement() is not { } why) return;
        if (outp["ending_disagree"] is not JsonArray list) outp["ending_disagree"] = list = new JsonArray();
        list.Add(new JsonObject { ["step"] = step, ["readings"] = why });
    }

    static JsonObject Step(IEngine e, JsonObject? action, bool fields)
    {
        var step = new JsonObject { ["action"] = action, ["state_hash"] = e.StateHash() };
        if (fields)
        {
            var o = new JsonObject();
            foreach (var (k, v) in e.StateFieldHashes()) o[k] = v;
            step["state_fields"] = o;
        }

        return step;
    }

    /// <summary>Answers every screen open, one pick (or the decline) at a time, through the replay's own ApplyChoice; each answer is logged as the referee logs one.</summary>
    static void Answer(IEngine e, int step, Mix rng, double pDecline, JsonArray choices)
    {
        for (int guard = 0; e.Screen is { } sel; guard++)
        {
            if (guard >= 64) throw new InvalidOperationException("a screen stayed open through 64 answers: " + sel.Kind);
            bool decline = sel.Skippable && (sel.Candidates.Count == 0 || rng.Chance(pDecline));
            if (!decline && sel.Candidates.Count == 0) throw new InvalidOperationException("an unskippable screen with no cards: " + sel.Kind);
            var picked = decline ? new JsonArray() : new JsonArray(rng.Next(sel.Candidates.Count));
            var choice = new JsonObject
            {
                ["kind"] = "cards", ["step"] = step, ["max"] = 1,
                ["options"] = new JsonArray(sel.Candidates.Select(c => (JsonNode)c.ModelId).ToArray()), ["picked"] = picked,
            };
            var log = Fight.ApplyChoice(e, choice);
            if (log["error"] != null) throw new InvalidOperationException("fuzz answer: " + log["error"]);
            choices.Add(choice);
        }
    }

    /// <summary>SplitMix64: the same draws on every runtime, so a spec's seed names its fight.</summary>
    sealed class Mix(ulong seed)
    {
        ulong _s = seed;

        ulong Next64()
        {
            ulong z = _s += 0x9E3779B97F4A7C15UL;
            z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9UL;
            z = (z ^ (z >> 27)) * 0x94D049BB133111EBUL;
            return z ^ (z >> 31);
        }

        internal int Next(int n) => n <= 1 ? 0 : (int)(Next64() % (ulong)n);

        internal bool Chance(double p) => (Next64() >> 11) * (1.0 / (1UL << 53)) < p;
    }
}
