// EmuDump fight --rng-log: every draw from a combat stream, per step, as [stream, index, caller].
// tools/fidelity/rng_diff.py diffs two builds' logs fight by fight, so a refactor that keeps every
// stream's draw COUNT but moves a draw to another caller (or another step) is caught; a count-only
// diff cannot see that.
//
// index is the stream's counter after the draw; caller is the first two engine frames above the
// stream (as trace mode names them: "CardEffects.ShufflePile < CombatEngine.EndTurn"), with lambda and
// local-function suffixes dropped so a new lambda elsewhere in a method does not rename its draws.
// The JIT inlines small methods away from the stack, and how it inlines can change with any edit, so
// rng_diff.py runs EmuDump with DOTNET_TieredCompilation=0 and DOTNET_JITMinOpts=1 (no inlining).
//
// A screen raised in the middle of a step stops the step, and its answer runs the step again from its
// start (CombatEngine.Step, HookChoiceReplay), with the streams moved back to where the step found them.
// The replay draws the stopped run's draws again, so at the replay's mark ("replay") every draw logged
// past its stream's position now is dropped -- from this step's draws, and from a step already written
// when the screen was answered a step later -- as trace mode drops the stopped run's events (Trace.cs):
// the log holds each draw once, as the game made it.
using System.Diagnostics;
using System.Text.Json.Nodes;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;

static class RngLog
{
    /// <summary>A draw: its stream, the stream's counter after it, its caller, and the step array it went to (null while pending).</summary>
    sealed class Draw(CountingRandom stream, int index, string caller)
    {
        public readonly CountingRandom Stream = stream;
        public readonly int Index = index;
        public readonly string Caller = caller;
        public JsonArray? WrittenIn;
        public JsonNode? Node;
    }

    /// <summary>The fight's draws so far, in order, the written ones too: a replay takes back a stopped run's.</summary>
    static readonly List<Draw> Logged = [];

    /// <summary>Starts logging every draw of every CountingRandom in this process.</summary>
    public static void Install()
    {
        CombatTrace.RngCall = r => Logged.Add(new Draw(r, r.CallCount, Caller()));
        var marked = CombatTrace.Mark;
        CombatTrace.Mark = (what, subject, state) =>
        {
            marked?.Invoke(what, subject, state);
            if (what == "replay") DropRewound();
        };
    }

    /// <summary>
    /// The replay of a stopped step has moved its streams back to the step's start: a draw past its
    /// stream's position is the stopped run's, which the replay draws again. Dropped, wherever it was written.
    /// </summary>
    static void DropRewound()
    {
        for (int i = Logged.Count - 1; i >= 0; i--)
        {
            var d = Logged[i];
            if (d.Index <= d.Stream.CallCount) continue;
            d.WrittenIn?.Remove(d.Node);
            Logged.RemoveAt(i);
        }
    }

    /// <summary>The draws since the last call, named by the streams of <paramref name="e"/>'s combat.</summary>
    public static JsonArray Take(RunEngine e)
    {
        var c = e.State.ActiveCombat;
        var arr = new JsonArray();
        foreach (var d in Logged)
        {
            if (d.WrittenIn != null) continue;
            d.Node = new JsonArray(Name(d.Stream, c, e), d.Index, d.Caller);
            arr.Add(d.Node);
            d.WrittenIn = arr;
        }

        return arr;
    }

    /// <summary>Forgets the fight's draws: a new fight, or one that threw before its first step.</summary>
    public static void Clear() => Logged.Clear();

    static string Name(CountingRandom r, CombatState? c, RunEngine e)
    {
        if (c != null)
        {
            if (ReferenceEquals(r, c.ShuffleRng)) return "Shuffle";
            if (ReferenceEquals(r, c.TargetRng)) return "CombatTargets";
            if (ReferenceEquals(r, c.AiRng)) return "MonsterAi";
            if (ReferenceEquals(r, c.CardSelectionRng)) return "CombatCardSelection";
            if (ReferenceEquals(r, c.CardGenerationRng)) return "CombatCardGeneration";
            if (ReferenceEquals(r, c.EnergyCostRng)) return "CombatEnergyCosts";
            if (ReferenceEquals(r, c.PotionGenerationRng)) return "CombatPotionGeneration";
            if (ReferenceEquals(r, c.OrbGenerationRng)) return "CombatOrbGeneration";
            if (ReferenceEquals(r, c.NicheHpRng)) return "Niche";
        }

        if (ReferenceEquals(r, e.State.ActiveCombatRng)) return "combat";  // the combat's own stream (trace mode's "fallback")
        return $"other:{(uint)r.Seed:x8}";  // a stream the combat no longer holds (replaced during the step), by its seed
    }

    static string Caller()
    {
        var parts = new List<string>(2);
        foreach (var f in new StackTrace(2, false).GetFrames())
        {
            var m = f.GetMethod();
            var t = m?.DeclaringType;
            if (m == null || t == null || t == typeof(CountingRandom) || t == typeof(RngLog) || t == typeof(CombatTrace)
                || (t.Namespace ?? "").StartsWith("System", StringComparison.Ordinal))
            {
                continue;
            }

            string mn = m.Name, tn = t.Name;
            if (mn.StartsWith('<') && mn.IndexOf('>') > 1) mn = mn[1..mn.IndexOf('>')];  // a lambda or local function: its method
            while (tn.StartsWith('<') && t.DeclaringType != null)
            {
                t = t.DeclaringType;
                tn = t.Name;
            }

            parts.Add(tn + "." + mn);
            if (parts.Count == 2) break;
        }

        return string.Join(" < ", parts);
    }
}
