// The fight solver's front end (docs/rebuild/FIGHT_SOLVER.md section 1, piece S4).
// Usage: EmuDump solve <ascension> <seed> [--at <step>] [--potions A,B] [--hp N] [options] > solve.jsonl
//        EmuDump solve --record <referee out.json> --decision <i> [options]
//        EmuDump solve --snapshot <doc.json | docs.jsonl> [--load-seed S] [options]   (one line per document)
//        EmuDump hindsight <ascension> <seed> [--at <step>] --allow-hindsight [--max-turns T] [options]
// options: --engine old|core2, --budget-nodes N, --budget-steps N, --budget-ms M, --horizon H, --prune all|none|<names,...>
//          (a name with a leading '-' switches that pruning off from all), --curve, --mode expected|hindsight.
//
// One JSON object per decision: the expected result under "expected" (with the potion curve under "curve" when asked) or
// the hindsight bound under "hindsight", never both in one object, plus counters. The seed form starts from the fight the
// native handle deals (NativeExports.DealFight: Sts2_Create's at the default level, Sts2_ResetArena's otherwise) and plays
// <step> seeded random legal moves through the native step's own gate; the record form re-anchors through the reanchor path
// (Reanchor.cs) up to the record's decision <i>; the snapshot form loads `sts2-emu-snapshot/1` documents as
// Sts2_LoadCombatSnapshot does (NativeExports.LoadFight), each line of a .jsonl file a document or an object with "doc"
// (and "id", copied to the output). Under `dotnet EmuDump.dll` this is the JIT side of the solver's AOT-against-JIT gate
// (tests/python/test_solver_aot_sites.py). All forms solve through Sts2Emulator.Solver.SolverService, the code the native
// exports run, so the two answer alike. Hindsight refuses without --allow-hindsight, and on a re-anchored record, whose
// streams are the emulator's and not the game's.
using System.Text.Json.Nodes;
using EmuDump.Engine;
using EmuDump.Engines;
using EmuDump.Engines.Old;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;
using Sts2Emulator.GeneratedData;
using Sts2Emulator.Interop;
using Sts2Emulator.Solver;

static class SolveMode
{
    /// <summary>The parsed command line.</summary>
    internal sealed record Args(
        string Mode, int Ascension, int Seed, int At, int[] Potions, int Hp, string? Record, int Decision,
        SolveOptions Options, bool Curve, bool AllowHindsight, int MaxTurns, string? Snapshot = null, int LoadSeed = 0);

    public static int Run(string command, string[] args, TextWriter output)
    {
        Args a;
        try
        {
            a = Parse(command, args);
        }
        catch (ArgumentException e)
        {
            Console.Error.WriteLine($"EmuDump {command}: {e.Message}");
            return 2;
        }

        if (a.Snapshot is { } file)
        {
            foreach (var (header, doc) in Documents(file))
            {
                output.WriteLine(Decision(a, header, () => SnapshotPosition(doc, a.LoadSeed)).ToJsonString());
            }
        }
        else
        {
            output.WriteLine(Decision(a).ToJsonString());
        }

        output.Flush();
        return 0;
    }

    internal static Args Parse(string command, string[] args)
    {
        string mode = command == "hindsight" ? "hindsight" : "expected";
        int asc = Ascension.DefaultLevel, seed = 0, at = 0, hp = 0, decision = 0, maxTurns = new HindsightOptions().MaxTurns;
        int[] potions = [];
        string? record = null, snapshot = null;
        int loadSeed = 0;
        var opts = new SolveOptions();
        bool curve = false, allow = false;
        var positional = new List<string>();
        for (int i = 0; i < args.Length; i++)
        {
            string Next() => i + 1 < args.Length ? args[++i] : throw new ArgumentException($"{args[i]} needs a value");
            switch (args[i])
            {
                case "--at": at = int.Parse(Next()); break;
                case "--potions": potions = [.. Next().Split(',', StringSplitOptions.RemoveEmptyEntries).Select(PotionId)]; break;
                case "--hp": hp = int.Parse(Next()); break;
                case "--record": record = Next(); break;
                case "--decision": decision = int.Parse(Next()); break;
                case "--snapshot": snapshot = Next(); break;
                case "--load-seed": loadSeed = int.Parse(Next()); break;
                case "--budget-nodes": opts = opts with { NodeBudget = long.Parse(Next()) }; break;
                case "--budget-steps": opts = opts with { StepBudget = long.Parse(Next()) }; break;
                case "--budget-ms": opts = opts with { TimeBudgetMs = int.Parse(Next()) }; break;
                case "--horizon": opts = opts with { MaxHorizon = int.Parse(Next()) }; break;
                case "--prune": opts = opts with { Prune = Prune(Next()) }; break;
                case "--no-estimate": opts = opts with { UseEstimate = false }; break;
                case "--curve": curve = true; break;
                case "--allow-hindsight": allow = true; break;
                case "--max-turns": maxTurns = int.Parse(Next()); break;
                case "--mode":
                    mode = Next();
                    if (mode is not ("expected" or "hindsight")) throw new ArgumentException("--mode needs expected or hindsight");
                    break;
                default:
                    if (args[i].StartsWith("--", StringComparison.Ordinal)) throw new ArgumentException($"unknown option {args[i]}");
                    positional.Add(args[i]);
                    break;
            }
        }

        if (snapshot is not null)
        {
            if (positional.Count != 0 || record is not null) throw new ArgumentException("--snapshot takes no <ascension> <seed> and no --record");
        }
        else if (record is null)
        {
            if (positional.Count != 2) throw new ArgumentException("needs <ascension> <seed>, or --record <file> --decision <i>");
            asc = int.Parse(positional[0]);
            seed = int.Parse(positional[1]);
        }
        else if (positional.Count != 0)
        {
            throw new ArgumentException("--record takes no <ascension> <seed>");
        }

        if (curve && mode == "hindsight") throw new ArgumentException("--curve is an expected-mode output");
        return new Args(mode, asc, seed, at, potions, hp, record, decision, opts, curve, allow, maxTurns, snapshot, loadSeed);
    }

    static int PotionId(string name) =>
        int.TryParse(name, out int id) ? id : Potions.FindId(name) ?? throw new ArgumentException($"unknown potion {name}");

    internal static Pruning Prune(string spec)
    {
        if (spec == "all") return Pruning.All;
        if (spec == "none") return Pruning.None;
        var p = spec.StartsWith('-') ? Pruning.All : Pruning.None;
        foreach (var part in spec.Split(',', StringSplitOptions.RemoveEmptyEntries))
        {
            bool off = part.StartsWith('-');
            if (!Enum.TryParse<Pruning>(off ? part[1..] : part, out var bit) || bit is Pruning.None or Pruning.All)
            {
                throw new ArgumentException($"unknown pruning {part}");
            }

            p = off ? p & ~bit : p | bit;
        }

        return p;
    }

    /// <summary>The seed form's position: the native handle's fight and <c>at</c> seeded random moves through the native step's gate.</summary>
    internal static (CombatState State, CountingRandom Rng, string? Error) SeedPosition(Args a)
    {
        var kernel = EngineFactory.Selected == "core2" ? NativeEngine.Parse("core2").Kernel : null;
        var (state, rng) = NativeExports.DealFight(a.Seed, a.Ascension, kernel, a.Potions, a.Hp);
        var policy = new Random(a.Seed);
        for (int i = 0; i < a.At; i++)
        {
            int[] valid = [.. CombatEngine.ValidActions(state).Order()];  // ascending, as a C caller reads Sts2_ValidActions
            var r = KernelGate.Step(kernel, state, rng, valid[policy.Next(valid.Length)], -1);
            if (r.Terminal)
            {
                return (state, rng, $"the fight ended at step {i + 1}, before step {a.At}");
            }
        }

        return (state, rng, null);
    }

    /// <summary>The record form's position: the reanchor path's fight after the record's first <c>decision</c> steps.</summary>
    internal static (CombatState? State, CountingRandom? Rng, string? Error) RecordPosition(Args a)
    {
        var game = JsonNode.Parse(File.ReadAllText(a.Record!))!.AsObject();
        var report = new JsonObject();
        IEngine e = EngineFactory.Create();
        e.StartFight(new FightSpec(game), report);
        var opening = e.ReanchorOpen();
        var gsteps = game["steps"]!.AsArray();
        var choices = (game["choices"]?.AsArray() ?? new JsonArray()).ToList();
        var untagged = Fight.UntaggedChoices.Of(game);
        Func<IScreenSource, JsonArray, List<JsonObject>>? answerOpen = untagged == null ? null : (src, log) => untagged.AnswerOpen(src, log);
        var carry = opening.Carry;
        var mode = new ReanchorMode(false, null, false, false);
        if (a.Decision >= gsteps.Count)
        {
            return (null, null, $"the record has {gsteps.Count - 1} steps");
        }

        for (int j = 1; j <= a.Decision; j++)
        {
            var rec = new JsonObject { ["action"] = gsteps[j]!["action"]!.AsObject().DeepClone() };
            var outcome = e.ReanchorStep(carry, mode, new ReanchorInput(gsteps, j, choices, answerOpen), rec);
            if (outcome.Stop)
            {
                return (null, null, $"re-anchoring stopped at step {j}");
            }

            carry = outcome.Carry;
        }

        var run = OldEngine.Of(e).Run.State;
        return run.ActiveCombat is { } c && run.ActiveCombatRng is { } r ? (c, r, null) : (null, null, "no fight is open at that decision");
    }

    /// <summary>The documents of a snapshot file: one JSON document, or one per line (each bare or under "doc").</summary>
    internal static IEnumerable<(JsonObject Header, JsonObject Doc)> Documents(string file)
    {
        string text = File.ReadAllText(file);
        var lines = file.EndsWith(".jsonl", StringComparison.OrdinalIgnoreCase)
            ? text.Split('\n').Where(l => l.Trim().Length > 0)
            : new[] { text };
        foreach (var line in lines)
        {
            var o = JsonNode.Parse(line)!.AsObject();
            var header = new JsonObject();
            if (o["doc"] is JsonObject doc)
            {
                foreach (var key in new[] { "id", "name" })
                {
                    if (o[key] is { } v)
                    {
                        header[key] = v.DeepClone();
                    }
                }

                yield return (header, doc);
            }
            else
            {
                yield return (header, o);
            }
        }
    }

    /// <summary>The snapshot form's position: the document read and seeded as <c>Sts2_LoadCombatSnapshot</c> reads it.</summary>
    internal static (CombatState? State, CountingRandom? Rng, string? Error) SnapshotPosition(JsonObject doc, int seed)
    {
        var kernel = EngineFactory.Selected == "core2" ? NativeEngine.Parse("core2").Kernel : null;
        var (state, rng) = NativeExports.LoadFight(System.Text.Encoding.UTF8.GetBytes(doc.ToJsonString()), seed, kernel);
        return (state, rng, null);
    }

    internal static JsonObject Decision(Args a) =>
        Decision(a, new JsonObject(), () => a.Record is null ? SeedPosition(a) : RecordPosition(a));

    internal static JsonObject Decision(Args a, JsonObject header, Func<(CombatState? State, CountingRandom? Rng, string? Error)> position)
    {
        var o = new JsonObject
        {
            ["solver_version"] = SolverService.Version,
            ["engine"] = EngineFactory.Selected,
            ["mode"] = a.Mode,
        };
        foreach (var (k, v) in header)
        {
            o[k] = v?.DeepClone();
        }

        if (a.Snapshot is not null)
        {
            o["load_seed"] = a.LoadSeed;
        }
        else if (a.Record is null)
        {
            o["ascension"] = a.Ascension;
            o["seed"] = a.Seed;
            o["at"] = a.At;
        }
        else
        {
            o["decision"] = a.Decision;
        }

        CombatState? state;
        CountingRandom? rng;
        string? error;
        try
        {
            (state, rng, error) = position();
        }
        catch (Exception ex) when (ex is not OutOfMemoryException)
        {
            (state, rng, error) = (null, null, $"{ex.GetType().Name}: {ex.Message}");
        }

        if (error is not null || state is null || rng is null)
        {
            o["error"] = error ?? "no position";
            return o;
        }

        o["engine"] = SolverService.EngineOf(state);
        if (a.Mode == "hindsight")
        {
            var h = SolverService.SolveHindsight(state, rng, new HindsightOptions
            {
                AllowHindsight = a.AllowHindsight,
                SearchNodeBudget = a.Options.NodeBudget,
                SearchStepBudget = a.Options.StepBudget,
                SearchTimeMs = a.Options.TimeBudgetMs,
                MaxTurns = a.MaxTurns,
            }, trueStreams: a.Record is null && a.Snapshot is null);
            o["hindsight"] = SolverService.HindsightJson(h);
            o["counters"] = new JsonObject { ["nodes"] = h.NodesSearched, ["steps"] = h.EngineSteps, ["seconds"] = h.Elapsed };
            return o;
        }

        var r = SolverService.SolveExpected(state, rng, a.Options);
        o["expected"] = SolverService.ExpectedJson(state, r);
        o["counters"] = SolverService.CountersJson(r.Counters, r.Seconds);
        if (a.Curve)
        {
            o["curve"] = SolverService.CurveJson(SolverService.SolvePotionCurve(state, rng, a.Options), state);
        }

        return o;
    }
}
