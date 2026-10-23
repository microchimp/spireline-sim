// The old engine's side of EmuDump fight --rng-log (IEngine.StartRngLog, TakeRngLog): every draw from a combat stream,
// per step, with the game's stream name, the stream's counter after the draw and the caller. tools/emudump/RngLog.cs
// shapes the draws into [stream, index, caller] arrays and tools/fidelity/rng_diff.py diffs two builds' logs fight by
// fight, so a refactor that keeps every stream's draw COUNT but moves a draw to another caller (or another step) is
// caught; a count-only diff cannot see that.
//
// index is the stream's counter after the draw; caller is the first two engine frames above the stream (as trace
// mode names them: "CardEffects.ShufflePile < CombatEngine.EndTurn"), with lambda and local-function suffixes
// dropped so a new lambda elsewhere in a method does not rename its draws. The JIT inlines small methods away from
// the stack, and how it inlines can change with any edit, so every tool that reads the labels runs EmuDump with
// tools/fidelity/rng_diff.py's NO_INLINING environment (DOTNET_TieredCompilation=0, DOTNET_JITMinOpts=1).
//
// A screen raised in the middle of a step stops the step, and its answer runs the step again from its start
// (CombatEngine.Step, HookChoiceReplay), with the streams moved back to where the step found them. The replay draws
// the stopped run's draws again, so at the replay's mark ("replay") every draw logged past its stream's position now
// is dropped -- from this step's draws, and from a step already written when the screen was answered a step later
// (the draw's TakenBack tells whoever holds it) -- as trace mode drops the stopped run's events (OldTrace.cs): the log
// holds each draw once, as the game made it.
using System.Diagnostics;
using EmuDump.Engine;
using Sts2Emulator.Core;
using Sts2Emulator.Core.Run;

namespace EmuDump.Engines.Old;

static class OldRngLog
{
    /// <summary>A draw as the stream made it, and the draw handed out for it (null while pending).</summary>
    sealed class Entry(CountingRandom stream, int index, string caller)
    {
        public readonly CountingRandom Stream = stream;
        public readonly int Index = index;
        public readonly string Caller = caller;
        public RngDraw? Handed;
    }

    /// <summary>The fight's draws so far, in order, the handed-out ones too: a replay takes back a stopped run's.</summary>
    static readonly List<Entry> Logged = [];

    // The hooks installed, kept to tell whether they still are (a test puts the seams back when it is done).
    static Action<CountingRandom>? _rngHook;
    static Action<string, int, CombatState>? _markHook;

    /// <summary>Starts logging every draw of every CountingRandom in this process (the hooks are the process's), and forgets the draws logged so far.</summary>
    public static void Start()
    {
        Install();
        Logged.Clear();
    }

    /// <summary>Installs the hooks unless they are in place already.</summary>
    internal static void Install()
    {
        if (_rngHook != null && CombatTrace.RngCall == _rngHook && CombatTrace.Mark == _markHook) return;
        var marked = CombatTrace.Mark;
        _rngHook = r => Logged.Add(new Entry(r, r.CallCount, Caller()));
        _markHook = (what, subject, state) =>
        {
            marked?.Invoke(what, subject, state);
            if (what == "replay") DropRewound();
        };
        CombatTrace.RngCall = _rngHook;
        CombatTrace.Mark = _markHook;
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
            d.Handed?.TakenBack?.Invoke();
            Logged.RemoveAt(i);
        }
    }

    /// <summary>The draws since the last call, named by the streams of <paramref name="e"/>'s combat.</summary>
    public static IReadOnlyList<RngDraw> Take(RunEngine e)
    {
        var c = e.State.ActiveCombat;
        var taken = new List<RngDraw>();
        foreach (var d in Logged)
        {
            if (d.Handed != null) continue;
            d.Handed = new RngDraw(Name(d.Stream, c, e), d.Index, d.Caller);
            taken.Add(d.Handed);
        }

        return taken;
    }

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
            if (m == null || t == null || t == typeof(CountingRandom) || t == typeof(OldRngLog) || t == typeof(CombatTrace)
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
