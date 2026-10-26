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
                if ((long)Max <= Min)
                {
                    // The stream throws on an empty range (MegaRandom.Next): no outcome, and never a
                    // node whose leaves lose its probability.
                    throw new ChanceRefused("empty-range", this);
                }

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
    /// <paramref name="source" />; null unhooks them. A combat with a stream unset (or a plain Random)
    /// refuses a source (<c>null-stream</c>): the engine's <c>?? new Random(0)</c> fallbacks would roll
    /// there unseen by the hook, and the fight would be solved silently wrong.
    /// </summary>
    public static void Attach(CombatState state, Random? stepRng, IChanceSource? source)
    {
        if (source is not null)
        {
            if (stepRng is not null and not CountingRandom)
            {
                throw new ChanceRefused("null-stream", new ChanceCall(-1, 0, ChanceKind.Next, 0, 0, null, null));
            }

            for (int i = 0; i < StateCloning.StreamCount; i++)
            {
                if (StateCloning.Stream(state, i) is null)
                {
                    throw new ChanceRefused("null-stream", new ChanceCall(i, 0, ChanceKind.Next, 0, 0, null, null));
                }
            }
        }

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

/// <summary>An answer a chance source gave inside a step: the stream, the call's index on it, the value.</summary>
public readonly record struct ChanceAnswer(int Stream, int Index, int Value);

/// <summary>
/// A stopped step's chance answers given again on its replay (a screen's answer runs the step again from
/// its start, <c>CombatEngine.Step</c>): each recorded answer, in order, to the call at its stream and
/// index; every other call goes to the source the stream had (<see cref="Inner" />), or draws. So a
/// step a source steered to a screen replays the same way after the source is detached, on the
/// combat's own streams, and the source is never asked twice for one answer.
/// </summary>
internal sealed class RecordedChance(IReadOnlyList<ChanceAnswer> record, int[] cursor, IChanceSource? inner) : IChanceSource
{
    private bool _fromRecord;

    public IChanceSource? Inner { get; } = inner;

    public int? Answer(in ChanceCall call)
    {
        int n = cursor[0];
        if (n < record.Count && record[n].Stream == call.Stream && record[n].Index == call.Index && call.Kind != ChanceKind.RawDouble)
        {
            cursor[0] = n + 1;
            _fromRecord = true;
            return record[n].Value;
        }

        _fromRecord = false;
        return Inner?.Answer(in call);
    }

    public void Drawn(in ChanceCall call, int value)
    {
        if (!_fromRecord)
        {
            Inner?.Drawn(in call, value);
        }

        _fromRecord = false;
    }

    /// <summary>Puts <paramref name="record" /> in front of every stream of <paramref name="state" /> and <paramref name="stepRng" />, over each one's own source when <paramref name="keepInner" />.</summary>
    internal static void Install(CombatState state, Random? stepRng, IReadOnlyList<ChanceAnswer> record, bool keepInner)
    {
        var cursor = new int[1];
        if (stepRng is CountingRandom step)
        {
            step.SetChance(new RecordedChance(record, cursor, keepInner ? step.Chance : null), -1);
        }

        for (int i = 0; i < StateCloning.StreamCount; i++)
        {
            if (StateCloning.Stream(state, i) is { } s)
            {
                s.SetChance(new RecordedChance(record, cursor, keepInner ? s.Chance : null), i);
            }
        }
    }

    /// <summary>Takes the record off the streams in <paramref name="state" />'s slots and <paramref name="stepRng" />: each answers to its own source again.</summary>
    internal static void Remove(CombatState state, Random? stepRng)
    {
        if (stepRng is CountingRandom { Chance: RecordedChance rs } step)
        {
            step.SetChance(rs.Inner, step.ChanceStream);
        }

        for (int i = 0; i < StateCloning.StreamCount; i++)
        {
            if (StateCloning.Stream(state, i) is { Chance: RecordedChance r } s)
            {
                s.SetChance(r.Inner, s.ChanceStream);
            }
        }
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
