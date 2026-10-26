namespace Sts2Emulator.Solver;

using System.Diagnostics;
using Sts2Emulator.Core;

/// <summary>What a stream was asked for (docs/rebuild/FIGHT_SOLVER.md section 4).</summary>
public enum ChanceKind
{
    /// <summary><c>Next(max)</c>, <c>Next(min, max)</c> or <c>Next()</c>: uniform over [Min, Max).</summary>
    Next,

    /// <summary>A weighted branch through <c>Roll.Branch</c>: odds weight / sum.</summary>
    Branch,

    /// <summary>A raw <c>NextDouble</c>: no odds known, so a solve refuses it (<c>unlisted-roll</c>).</summary>
    RawDouble,
}

/// <summary>
/// One random call an engine step makes: the stream (its index in <c>StateCloning.StreamNames</c>,
/// -1 for the step's own stream), the call's index on that stream (its <c>CallCount</c> before the
/// call), and its range or branch weights.
/// </summary>
public readonly record struct ChanceCall(int Stream, int Index, ChanceKind Kind, int Min, int Max, double[]? Weights, string? Site)
{
    public string StreamName => Stream < 0 ? "combat" : StateCloning.StreamNames[Stream];

    /// <summary>The number of outcomes the call can have (positive weights for a branch).</summary>
    public long Width => Kind switch
    {
        ChanceKind.Next => (long)Max - Min,
        ChanceKind.Branch => Math.Max(1, Weights!.Count(w => w > 0)),
        _ => 0,
    };

    /// <summary>
    /// The call's outcomes at the rules' odds: each value of the range with 1 / width, or each branch
    /// with a positive weight with weight / sum (an all-zero branch list has the outcome 0 alone).
    /// </summary>
    public IEnumerable<(int Value, double P)> Outcomes()
    {
        switch (Kind)
        {
            case ChanceKind.Next:
                double p = 1d / ((long)Max - Min);
                for (long v = Min; v < Max; v++)
                {
                    yield return ((int)v, p);
                }

                break;
            case ChanceKind.Branch:
                double sum = Weights!.Sum();
                if (sum <= 0)
                {
                    yield return (0, 1d);
                    yield break;
                }

                for (int i = 0; i < Weights.Length; i++)
                {
                    if (Weights[i] > 0)
                    {
                        yield return (i, Weights[i] / sum);
                    }
                }

                break;
            default:
                throw new ChanceRefused("unlisted-roll", this);
        }
    }

    public override string ToString() =>
        $"{StreamName}#{Index} {Kind}" + (Kind == ChanceKind.Branch ? $" {Site} [{string.Join(",", Weights!)}]" : $" [{Min},{Max})");
}

/// <summary>
/// The chance hook's other end: a stream hooked to a source puts each call to it. Outside the
/// solver no stream has one, and nothing changes.
/// </summary>
public interface IChanceSource
{
    /// <summary>
    /// The call's answer (a value in [Min, Max), or a branch index), or null to let the stream's own
    /// draw stand. May throw <see cref="ChancePending" /> to stop the step at the call.
    /// </summary>
    int? Answer(in ChanceCall call);

    /// <summary>The value the call returned (-1 for a raw double), after the stream took its draw.</summary>
    void Drawn(in ChanceCall call, int value)
    {
    }
}

/// <summary>A step stopped at a random call whose answer is not known yet: the twin of a screen's pending choice.</summary>
public sealed class ChancePending(ChanceCall call) : Exception($"chance pending at {call}")
{
    public ChanceCall Call { get; } = call;
}

/// <summary>A random call the solver cannot branch on as the rules wrote it, with the reason and the site.</summary>
public sealed class ChanceRefused(string reason, ChanceCall call) : Exception($"{reason}: {call} at {ChanceHook.Site()}")
{
    public string Reason { get; } = reason;

    public ChanceCall Call { get; } = call;
}

/// <summary>Hooks a combat's streams to a source, and the pass-through and replay sources.</summary>
public static class ChanceHook
{
    /// <summary>
    /// Hooks every stream of <paramref name="state" /> (and the step's own <paramref name="stepRng" />) to
    /// <paramref name="source" />; null unhooks them. A field that falls back to a plain Random is not
    /// a stream and stays as it is.
    /// </summary>
    public static void Attach(CombatState state, Random? stepRng, IChanceSource? source)
    {
        (stepRng as CountingRandom)?.SetChance(source, -1);
        for (int i = 0; i < StateCloning.StreamCount; i++)
        {
            StateCloning.Stream(state, i)?.SetChance(source, i);
        }
    }

    /// <summary>The first engine frame above the streams and the solver, for a refusal's message.</summary>
    internal static string Site()
    {
        foreach (var f in new StackTrace(1, false).GetFrames())
        {
            var t = f.GetMethod()?.DeclaringType;
            if (t is null || t == typeof(CountingRandom) || t == typeof(Core.Rng.Roll) || (t.Namespace ?? "").StartsWith("Sts2Emulator.Solver", StringComparison.Ordinal)
                || (t.Namespace ?? "").StartsWith("System", StringComparison.Ordinal))
            {
                continue;
            }

            while (t.DeclaringType is not null && t.Name.StartsWith('<'))
            {
                t = t.DeclaringType;
            }

            return t.Name + "." + f.GetMethod()!.Name;
        }

        return "?";
    }
}

/// <summary>Sees every call and lets each stream answer it: the hook on, the bytes unchanged.</summary>
public sealed class ChanceRecorder : IChanceSource
{
    public List<(ChanceCall Call, int Value)> Calls { get; } = [];

    public int? Answer(in ChanceCall call) => null;

    public void Drawn(in ChanceCall call, int value) => Calls.Add((call, value));
}

/// <summary>
/// Answers a step's calls, in call order, from a list of answers; the first call past the list stops
/// the step (<see cref="ChancePending" />). A raw double refuses (<c>unlisted-roll</c>), and so does a
/// call on a stream outside <see cref="Streams" /> unless <see cref="PassOthers" /> lets it draw.
/// </summary>
public sealed class ChanceReplay(IReadOnlyList<int> answers) : IChanceSource
{
    private int _next;

    /// <summary>The streams answered from the list (by index; -1 the step's own); null for all.</summary>
    public HashSet<int>? Streams { get; init; }

    /// <summary>Whether a call on a stream outside <see cref="Streams" /> takes its stream's own draw instead of refusing.</summary>
    public bool PassOthers { get; init; }

    /// <summary>Calls answered so far.</summary>
    public int Answered => _next;

    public int? Answer(in ChanceCall call)
    {
        if (Streams is not null && !Streams.Contains(call.Stream))
        {
            return PassOthers ? null : throw new ChanceRefused("unbranched-stream", call);
        }

        if (call.Kind == ChanceKind.RawDouble)
        {
            throw new ChanceRefused("unlisted-roll", call);
        }

        if (_next < answers.Count)
        {
            return answers[_next++];
        }

        throw new ChancePending(call);
    }
}

/// <summary>One leaf of a chance enumeration: what the run returned, its probability and the answers that reached it.</summary>
public readonly record struct ChanceLeaf<T>(T Result, double P, int[] Answers);

/// <summary>
/// Enumerates every outcome of a run whose random calls go through a hooked source: the run is
/// replayed from its start with each prefix of answers (the replay a stopped step takes), and each
/// pending call branches at the rules' odds.
/// </summary>
public static class ChanceTree
{
    /// <param name="run">Runs once per replay on fresh copies, hooked to the source it is given.</param>
    /// <param name="maxWidth">A pending call with more outcomes refuses (<c>chance-width</c>).</param>
    /// <param name="maxLeaves">More leaves than this refuses (<c>chance-leaves</c>).</param>
    public static List<ChanceLeaf<T>> Enumerate<T>(
        Func<IChanceSource, T> run,
        int maxWidth = 32,
        int maxLeaves = 100_000,
        HashSet<int>? streams = null,
        bool passOthers = false
    )
    {
        var leaves = new List<ChanceLeaf<T>>();
        var stack = new Stack<(int[] Answers, double P)>();
        stack.Push(([], 1d));
        while (stack.Count > 0)
        {
            var (answers, p) = stack.Pop();
            var source = new ChanceReplay(answers) { Streams = streams, PassOthers = passOthers };
            try
            {
                var result = run(source);
                leaves.Add(new ChanceLeaf<T>(result, p, answers));
                if (leaves.Count > maxLeaves)
                {
                    throw new ChanceRefused("chance-leaves", default);
                }
            }
            catch (ChancePending pending)
            {
                if (pending.Call.Width > maxWidth)
                {
                    throw new ChanceRefused("chance-width", pending.Call);
                }

                foreach (var (value, q) in pending.Call.Outcomes().Reverse())
                {
                    stack.Push(([.. answers, value], p * q));
                }
            }
        }

        return leaves;
    }
}
