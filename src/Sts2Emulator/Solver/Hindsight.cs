namespace Sts2Emulator.Solver;

using System.Diagnostics;
using Sts2Emulator.Core;

/// <summary>Whether a winning line exists given the record's draws and rolls (FIGHT_SOLVER.md section 9).</summary>
public enum WinLine
{
    /// <summary>The search was cut (budget or turn cap) before it found a win: neither yes nor no.</summary>
    Unknown = -1,

    No = 0,

    Yes = 1,
}

/// <summary>How a hindsight search ended (the native status codes of <c>Sts2_SolveHindsight</c>).</summary>
public enum HindsightOutcome
{
    /// <summary>Every line was searched to the fight's end: the best utility is the bound.</summary>
    Searched = 0,

    /// <summary>The budget or the turn cap cut the search: the best utility is the best line found, a lower bound on the bound.</summary>
    Cut = 1,

    /// <summary>Not searched: the flag was not set, the streams are not the record's, or the position was refused.</summary>
    Refused = -3,
}

/// <summary>
/// The options of a hindsight search. <see cref="AllowHindsight" /> is off by default: a search without it is refused, so
/// no caller reaches the bound by leaving a field unset (section 4a: only lost-fight attribution sets it).
/// </summary>
public sealed record HindsightOptions
{
    public bool AllowHindsight { get; init; }

    public long SearchNodeBudget { get; init; } = 2_000_000;

    public long SearchStepBudget { get; init; } = 20_000_000;

    /// <summary>Wall time in ms; 0 for none.</summary>
    public int SearchTimeMs { get; init; }

    /// <summary>Turns past the position's after which a line is cut (unknown, never a loss).</summary>
    public int MaxTurns { get; init; } = 30;
}

/// <summary>
/// The hindsight bound (<c>SolveHindsight</c>, FIGHT_SOLVER.md sections 4a and 9): the same fight with the record's draw
/// order and rolls fixed, an OR over lines. An upper bound on the losses better play could have avoided, read by lost-fight
/// attribution only: never a label, a regret or a grade. It shares no field with <see cref="ExpectedResult" />
/// (<c>HindsightApartTests</c>).
/// </summary>
public sealed record HindsightResult(
    HindsightOutcome Outcome,
    string? Refusal,
    WinLine WinLineExists,
    Rational BestUtility,
    bool Complete,
    IReadOnlyList<Move> Line,
    long NodesSearched,
    long EngineSteps,
    double Elapsed)
{
    public double BestUtilityDouble => BestUtility.ToDouble();
}

/// <summary>
/// The hindsight search: depth first over every legal move (each hand index and each target its own move: the multiset
/// prunings are the expected mode's), lethal-first order, each step run on a copy with the position's own streams and
/// no chance hook, so the draws and rolls are the record's. Positions are merged by the hindsight key (every stream's
/// seed and position and every pile in order, S1) plus the step's own stream; a line past the turn cap or the budget is
/// cut and counts as unknown. The best utility is L0 at the fight's end, the expected mode's terminal score.
/// </summary>
public sealed class Hindsight(HindsightOptions options)
{
    /// <summary>The refusal without the flag.</summary>
    public const string NotAllowed = "hindsight-not-allowed: the hindsight bound is for lost-fight attribution only, and the flag that allows it is not set";

    readonly Dictionary<UInt128, Entry> _memo = [];
    readonly HashSet<UInt128> _path = [];
    long _nodes;
    long _steps;
    bool _cut;
    int _turnCap;
    Stopwatch _sw = new();

    public HindsightOptions Options { get; } = options;

    readonly record struct Entry(Rational Best, bool Complete, Move BestMove, bool HasMove);

    public HindsightResult Solve(CombatState state, CountingRandom rng)
    {
        _sw = Stopwatch.StartNew();
        if (!Options.AllowHindsight)
        {
            return Refused(NotAllowed);
        }

        _memo.Clear();
        _path.Clear();
        (_nodes, _steps, _cut) = (0, 0, false);
        _turnCap = state.Turn + Math.Max(0, Options.MaxTurns);
        try
        {
            var root = Search(state, rng);
            var line = LineOf(state, rng);
            bool complete = root.Complete && !_cut;
            var verdict = root.Best > Utility.Loss ? WinLine.Yes : complete ? WinLine.No : WinLine.Unknown;
            return new HindsightResult(complete ? HindsightOutcome.Searched : HindsightOutcome.Cut, null, verdict, root.Best, complete, line,
                _nodes, _steps, _sw.Elapsed.TotalSeconds);
        }
        catch (SolveRefused e)
        {
            return Refused(e.Reason);
        }
    }

    HindsightResult Refused(string reason) =>
        new(HindsightOutcome.Refused, reason, WinLine.Unknown, Utility.Loss, false, [], _nodes, _steps, _sw.Elapsed.TotalSeconds);

    bool OverBudget() =>
        _nodes > Options.SearchNodeBudget || _steps > Options.SearchStepBudget
        || (Options.SearchTimeMs > 0 && _sw.ElapsedMilliseconds > Options.SearchTimeMs);

    /// <summary>The hindsight key and the step's own stream: the one stream the state does not hold.</summary>
    static UInt128 Key(CombatState s, CountingRandom rng)
    {
        byte[] state;
        try
        {
            state = StateKey.Encode(s, KeyMode.Hindsight);
        }
        catch (StateKeyRefusedException e)
        {
            throw new SolveRefused(e.Reason);
        }

        var bytes = new byte[state.Length + 8];
        state.CopyTo(bytes, 0);
        BitConverter.TryWriteBytes(bytes.AsSpan(state.Length), rng.Seed);
        BitConverter.TryWriteBytes(bytes.AsSpan(state.Length + 4), rng.CallCount);
        return StateKey.HashOf(bytes);
    }

    /// <summary>Lethal first: aimed plays, then the other plays, then potions and screen answers in order, end turn last.</summary>
    static List<Move> Ordered(CombatState s)
    {
        var moves = Actions.Legal(s, Pruning.None);
        if (s.PendingSelection is not null)
        {
            return moves;
        }

        int hand = s.Hand.Count;
        int Rank(Move m) => m.Action < hand ? (Actions.CardAims(s, s.Hand[m.Action]) ? 0 : 1) : m.Action == hand ? 3 : 2;
        return [.. moves.Select((m, i) => (m, i)).OrderBy(x => Rank(x.m)).ThenBy(x => x.i).Select(x => x.m)];
    }

    static StepResult StepCopy(CombatState s, CountingRandom rng, Move m, out CombatState copy, out CountingRandom r)
    {
        copy = s.Clone();
        r = rng.Copy();
        try
        {
            return Successors.Step(copy, r, m);
        }
        catch (Exception e) when (e is not OutOfMemoryException and not SolveRefused)
        {
            throw new SolveRefused($"engine-error: {e.GetType().Name}: {e.Message}");
        }
    }

    Entry Search(CombatState s, CountingRandom rng)
    {
        _nodes++;
        if (_cut || OverBudget())
        {
            _cut = true;
            return new Entry(Utility.Loss, false, default, false);
        }

        var key = Key(s, rng);
        if (_memo.TryGetValue(key, out var hit))
        {
            return hit;
        }

        if (!_path.Add(key))
        {
            throw new SolveRefused("cycle: a position reached again below itself");
        }

        Entry? best = null;
        bool complete = true;
        foreach (var m in Ordered(s))
        {
            _steps++;
            var result = StepCopy(s, rng, m, out var copy, out var r);
            Entry child = result.Terminal
                ? new Entry(Utility.Terminal(copy, result.PlayerWon), true, default, false)
                : copy.Turn > _turnCap
                    ? new Entry(Utility.Loss, false, default, false)
                    : Search(copy, r);
            complete &= child.Complete;
            if (best is null || child.Best > best.Value.Best)
            {
                best = new Entry(child.Best, false, m, true);
            }

            if (child.Best == Utility.Max)
            {
                // Nothing beats L0's maximum: the node's value is settled whatever its other moves hold.
                complete = true;
                break;
            }

            if (_cut)
            {
                complete = false;
                break;
            }
        }

        _path.Remove(key);
        var entry = best is { } b ? b with { Complete = complete } : new Entry(Utility.Loss, false, default, false);
        if (entry.Complete || !_cut)
        {
            _memo[key] = entry;
        }

        return entry;
    }

    /// <summary>The best line found, replayed from the position through the memo's best moves.</summary>
    List<Move> LineOf(CombatState state, CountingRandom rng)
    {
        var line = new List<Move>();
        var s = state;
        var r = rng;
        while (line.Count < 4096 && _memo.TryGetValue(Key(s, r), out var e) && e.HasMove)
        {
            line.Add(e.BestMove);
            var result = StepCopy(s, r, e.BestMove, out var copy, out var next);
            if (result.Terminal)
            {
                break;
            }

            (s, r) = (copy, next);
        }

        return line;
    }
}
