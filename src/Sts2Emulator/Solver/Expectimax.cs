namespace Sts2Emulator.Solver;

using System.Diagnostics;
using Sts2Emulator.Core;

/// <summary>How a solve ended (the native status codes of FIGHT_SOLVER.md section 1).</summary>
public enum SolveStatus
{
    /// <summary>Every leaf a fight's end: the value is the fight's expectimax value.</summary>
    Exact = 0,

    /// <summary>The budget, the size estimate or the horizon cap stopped the deepening: the last completed horizon, its cut valued by L0.</summary>
    Budget = 1,

    /// <summary>Not valued (a busy kernel queue, an unlisted roll, a hidden branch, no horizon within the budget): see the reason.</summary>
    Refused = -3,
}

public sealed record SolveOptions
{
    /// <summary>The deepest horizon tried: turns past the current one whose end turn is searched.</summary>
    public int MaxHorizon { get; init; } = 8;

    /// <summary>Decision nodes per solve (per decision, across its iterations).</summary>
    public long NodeBudget { get; init; } = 200_000;

    /// <summary>Engine step runs per solve.</summary>
    public long StepBudget { get; init; } = 2_000_000;

    /// <summary>Wall time per solve in ms; 0 for none.</summary>
    public int TimeBudgetMs { get; init; }

    public Pruning Prune { get; init; } = Pruning.All;

    /// <summary>A roll with more outcomes refuses (<c>chance-width</c>).</summary>
    public int ChanceWidthCap { get; init; } = 32;

    /// <summary>Skip a horizon whose size estimate is over the node budget by more than this many powers of ten.</summary>
    public double EstimateSlack { get; init; } = 1.0;

    public bool UseEstimate { get; init; } = true;
}

/// <summary>A root move's value.</summary>
public sealed record MoveValue(Move Move, Rational Value, bool Exact);

/// <summary>
/// The expected-mode result (<c>SolveExpected</c>): the value over the draw pile as a multiset and the rolls at the
/// game's odds. It shares no field with a hindsight result (FIGHT_SOLVER.md section 4a, S4).
/// </summary>
public sealed record ExpectedResult(
    SolveStatus Status,
    string? Reason,
    int Horizon,
    Rational Value,
    bool Exact,
    Rational ExactMass,
    IReadOnlyList<MoveValue> Moves,
    SolveCounters Counters,
    double Seconds)
{
    public double ValueDouble => Value.ToDouble();

    /// <summary>The moves of greatest value.</summary>
    public IEnumerable<Move> Best => Moves.Count == 0 ? [] : Moves.Where(m => m.Value == Moves.Max(x => x.Value)).Select(m => m.Move);
}

/// <summary>
/// A node's value, whether it is the fight's exact value, the probability mass that reached fights' ends, and a sound
/// upper bound on the exact value (a horizon leaf's is L0's maximum, 1). A max node is exact when a move of greatest
/// value is exact and no move's bound is above it: the moves cut at the horizon cannot beat it, whatever lies past
/// the cut. A chance node is exact when every outcome is.
/// </summary>
internal readonly record struct NodeValue(Rational Value, bool Exact, Rational Mass, Rational Upper)
{
    public static NodeValue Horizon(CombatState s) => new(Utility.L0(s), false, Rational.Zero, Utility.Max);

    public static NodeValue End(CombatState s, bool won)
    {
        var u = Utility.Terminal(s, won);
        return new(u, true, Rational.One, u);
    }

    /// <summary>The max node over its moves' values (in move order; the first of greatest value gives the mass).</summary>
    public static NodeValue Max(IEnumerable<NodeValue> moves, CombatState s)
    {
        NodeValue? best = null;
        Rational upper = Rational.Zero;
        bool any = false;
        var list = moves.ToList();
        foreach (var v in list)
        {
            if (best is null || v.Value > best.Value.Value)
            {
                best = v;
            }

            upper = !any || v.Upper > upper ? v.Upper : upper;
            any = true;
        }

        if (best is not { } b)
        {
            return Horizon(s);
        }

        bool exact = upper == b.Value && list.Exists(v => v.Exact && v.Value == b.Value);
        if (exact)
        {
            // The mass is the exact move's (all of it): the first move of greatest value may be a cut one whose bound is
            // its value (a horizon leaf at L0's maximum), whose mass is short of one.
            b = list.First(v => v.Exact && v.Value == b.Value);
        }

        return new NodeValue(b.Value, exact, b.Mass, exact ? b.Value : upper);
    }
}

/// <summary>
/// The exact layer (FIGHT_SOLVER.md sections 1, 5 and 7): expectimax over S1's expected-mode key with S2's chance,
/// max nodes over the legal moves (plays with their targets, potions, end turn, screen answers), chance nodes over
/// every roll and every draw from the unknown middle at the rules' odds, terminal leaves scored by L0 and the turn
/// horizon's end turns scored by L0 of the position as a placeholder for the network (S5). Values are exact
/// rationals. The table lives in the solver for the fight (subtree reuse). The position given is never changed:
/// every step runs on a copy.
/// </summary>
public sealed class Expectimax(SolveOptions? options = null)
{
    readonly HashSet<UInt128> _path = [];
    SolveCounters _c = new();

    public SolveOptions Options { get; } = options ?? new SolveOptions();

    public TranspositionTable Table { get; } = new();

    bool On(Pruning p) => (Options.Prune & p) != 0;

    static string EngineOf(CombatState s) =>
        s.Kernel is { AnyOn: true } k ? "core2:" + System.Runtime.CompilerServices.RuntimeHelpers.GetHashCode(k.Shared) : "old";

    /// <summary>Iterative deepening over the turn horizon, within the budget.</summary>
    public ExpectedResult SolveExpected(CombatState state, CountingRandom rng)
    {
        var sw = Stopwatch.StartNew();
        var counters = Begin(sw);
        ExpectedResult? last = null;
        string reason = "horizon";
        for (int h = 0; h <= Options.MaxHorizon; h++)
        {
            if (Options.UseEstimate && h > 0 && SizeEstimate.Log10(state, h) > Math.Log10(Options.NodeBudget) + Options.EstimateSlack)
            {
                reason = $"estimate: horizon {h} is about 10^{SizeEstimate.Log10(state, h):0.0} states";
                break;
            }

            try
            {
                last = Iterate(state, rng, h, counters, sw);
            }
            catch (BudgetExceeded e)
            {
                reason = e.Message;
                break;
            }
            catch (SolveRefused e)
            {
                return Refused(e.Reason, counters, sw);
            }

            if (last.Exact)
            {
                return last;
            }
        }

        return last is null ? Refused($"budget: no horizon completed ({reason})", counters, sw) : last with { Status = SolveStatus.Budget, Reason = reason, Seconds = sw.Elapsed.TotalSeconds };
    }

    /// <summary>One solve at a fixed horizon (the harness's call); refused or budget-cut solves come back as such.</summary>
    public ExpectedResult SolveAt(CombatState state, CountingRandom rng, int horizon)
    {
        var sw = Stopwatch.StartNew();
        var counters = Begin(sw);
        try
        {
            return Iterate(state, rng, horizon, counters, sw);
        }
        catch (BudgetExceeded e)
        {
            return Refused($"budget: {e.What}", counters, sw);
        }
        catch (SolveRefused e)
        {
            return Refused(e.Reason, counters, sw);
        }
    }

    SolveCounters Begin(Stopwatch sw)
    {
        if (!On(Pruning.SubtreeReuse))
        {
            Table.Clear();
        }

        Table.NextGeneration();
        var c = new SolveCounters();
        c.Check = () =>
        {
            if (c.Nodes > Options.NodeBudget)
            {
                throw new BudgetExceeded($"{Options.NodeBudget} nodes");
            }

            if (c.Steps > Options.StepBudget)
            {
                throw new BudgetExceeded($"{Options.StepBudget} steps");
            }

            if (Options.TimeBudgetMs > 0 && sw.ElapsedMilliseconds > Options.TimeBudgetMs)
            {
                throw new BudgetExceeded($"{Options.TimeBudgetMs} ms");
            }
        };
        return c;
    }

    ExpectedResult Refused(string reason, SolveCounters c, Stopwatch sw) =>
        new(SolveStatus.Refused, reason, -1, Rational.Zero, false, Rational.Zero, [], c, sw.Elapsed.TotalSeconds);

    ExpectedResult Iterate(CombatState state, CountingRandom rng, int horizon, SolveCounters counters, Stopwatch sw)
    {
        _c = counters;
        _path.Clear();
        Table.Bind(EngineOf(state));
        var key = Key(state);
        _path.Add(key);
        var moves = new List<MoveValue>();
        var values = new List<NodeValue>();
        foreach (var m in Actions.Legal(state, Options.Prune))
        {
            var v = MoveValueOf(state, rng, m, horizon);
            moves.Add(new MoveValue(m, v.Value, v.Exact));
            values.Add(v);
        }

        _path.Remove(key);
        var b = NodeValue.Max(values, state);
        bool exact = b.Exact && b.Mass == Rational.One;
        Table.Put(key, b, horizon);
        return new ExpectedResult(exact ? SolveStatus.Exact : SolveStatus.Budget, null, horizon, b.Value, exact, b.Mass, moves, counters, sw.Elapsed.TotalSeconds);
    }

    static UInt128 Key(CombatState s)
    {
        try
        {
            return StateKey.Hash(s, KeyMode.Expected);
        }
        catch (StateKeyRefusedException e)
        {
            throw new SolveRefused(e.Reason);
        }
    }

    NodeValue Node(CombatState s, CountingRandom rng, int remaining)
    {
        _c.Nodes++;
        _c.Check?.Invoke();
        var key = Key(s);
        if (On(Pruning.Transposition) && Table.TryGet(key, remaining, out var hit))
        {
            _c.TableHits++;
            return hit.Node;
        }

        if (s.PendingSelection is not null)
        {
            _c.Screens++;
        }

        if (!_path.Add(key))
        {
            throw new SolveRefused("cycle: a position reached again below itself");
        }

        var values = new List<NodeValue>();
        foreach (var m in Actions.Legal(s, Options.Prune))
        {
            values.Add(MoveValueOf(s, rng, m, remaining));
        }

        _path.Remove(key);
        var result = NodeValue.Max(values, s);
        if (On(Pruning.Transposition))
        {
            Table.Put(key, result, remaining);
        }

        return result;
    }

    NodeValue MoveValueOf(CombatState s, CountingRandom rng, Move m, int remaining)
    {
        if (Actions.EndsTurn(s, m) && remaining <= 0)
        {
            _c.HorizonLeaves++;
            return NodeValue.Horizon(s);
        }

        var outcomes = Successors.Expand(s, rng, m, byCard: !On(Pruning.DrawByValue), Options.ChanceWidthCap, _c);
        var value = Rational.Zero;
        var mass = Rational.Zero;
        var upper = Rational.Zero;
        bool exact = true;
        if (On(Pruning.Transposition))
        {
            // Equal outcomes are one child: terminal ones by their score, the rest by key.
            var groups = new Dictionary<UInt128, (Outcome First, Rational P)>();
            var order = new List<UInt128>();
            foreach (var o in outcomes)
            {
                if (o.Result.Terminal)
                {
                    _c.Leaves++;
                    var t = Utility.Terminal(o.State, o.Result.PlayerWon);
                    value += o.P * t;
                    upper += o.P * t;
                    mass += o.P;
                    continue;
                }

                var k = Key(o.State);
                if (groups.TryGetValue(k, out var g))
                {
                    _c.Merged++;
                    groups[k] = (g.First, g.P + o.P);
                }
                else
                {
                    groups[k] = (o, o.P);
                    order.Add(k);
                }
            }

            foreach (var k in order)
            {
                var (o, p) = groups[k];
                var v = Node(o.State, o.Rng, remaining - (o.State.Turn - s.Turn));
                value += p * v.Value;
                mass += p * v.Mass;
                upper += p * v.Upper;
                exact &= v.Exact;
            }
        }
        else
        {
            foreach (var o in outcomes)
            {
                if (o.Result.Terminal)
                {
                    _c.Leaves++;
                    var t = Utility.Terminal(o.State, o.Result.PlayerWon);
                    value += o.P * t;
                    upper += o.P * t;
                    mass += o.P;
                    continue;
                }

                var v = Node(o.State, o.Rng, remaining - (o.State.Turn - s.Turn));
                value += o.P * v.Value;
                mass += o.P * v.Mass;
                upper += o.P * v.Upper;
                exact &= v.Exact;
            }
        }

        return new NodeValue(value, exact, mass, exact ? value : upper);
    }
}
