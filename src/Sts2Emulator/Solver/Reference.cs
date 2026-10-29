namespace Sts2Emulator.Solver;

using Sts2Emulator.Core;

/// <summary>
/// The reference solver (FIGHT_SOLVER.md section 6): plain tree expectimax over full labelled states for small fights.
/// No table, no pruning: every hand index and potion slot is its own move, every targetable enemy its own target, a
/// draw from the unknown middle branches over each card at 1 / size, and equal outcomes are separate children.
/// What it shares with <see cref="Expectimax" />: <see cref="Successors.Expand" /> (the engine step, the roll odds,
/// the draw pile's layout and its branches, by card here, and every refusal it raises), <see cref="Actions.Legal" />
/// (under <see cref="Pruning.None" />), <see cref="NodeValue" /> (the max, the horizon leaf, the exactness rule),
/// <see cref="Utility" /> (L0 and the terminal score) and the key's refusals (<see cref="StateKey.Encode" />). It shares
/// no table, merging or pruning, so the equal-value harness holds each pruning to it, to the last bit (both compute
/// exact rationals); it cannot see a wrong chance model, which both would share: ChanceModelTests holds Expand to an
/// independent count over the engine's own seeded streams.
/// </summary>
public static class Reference
{
    public sealed record Result(Rational Value, bool Exact, IReadOnlyList<MoveValue> Moves, long Nodes, long Steps, string? Refused);

    /// <param name="tightBound">Horizon leaves and wide rolls carry the tight utility bound (<see cref="UtilityGains.Upper" />)
    /// where no gain is reachable, as the pruned solver's with <see cref="Pruning.UtilityBound" />: the same full tree, so
    /// the closure and the cuts are held to it.</param>
    /// <param name="utility">The utility (<see cref="UtilityKind" />): a check utility's leaves and ends, never with the tight bound.</param>
    public static Result Solve(CombatState state, CountingRandom rng, int horizon, long nodeCap = 2_000_000, int widthCap = 32, bool tightBound = false,
        UtilityKind utility = UtilityKind.L0)
    {
        var c = new SolveCounters();
        c.Check = () =>
        {
            if (c.Nodes > nodeCap || c.Steps > nodeCap * 4)
            {
                throw new BudgetExceeded($"reference cap {nodeCap}");
            }
        };
        try
        {
            StateKey.Encode(state, KeyMode.Expected);
            var moves = new List<MoveValue>();
            var values = new List<NodeValue>();
            foreach (var m in Actions.Legal(state, Pruning.None))
            {
                var v = MoveValueOf(state, rng, m, horizon, c, widthCap, tightBound, utility);
                moves.Add(new MoveValue(m, v.Value, v.Exact));
                values.Add(v);
            }

            var b = NodeValue.Max(values, state, utility);
            bool exact = utility != UtilityKind.L0 ? c.WideRollLeaves == 0 : b.Exact && b.Mass == Rational.One;
            return new Result(b.Value, exact, moves, c.Nodes, c.Steps, null);
        }
        catch (SolveRefused e)
        {
            return new Result(Rational.Zero, false, [], c.Nodes, c.Steps, e.Reason);
        }
        catch (StateKeyRefusedException e)
        {
            return new Result(Rational.Zero, false, [], c.Nodes, c.Steps, e.Reason);
        }
        catch (BudgetExceeded e)
        {
            return new Result(Rational.Zero, false, [], c.Nodes, c.Steps, "budget: " + e.What);
        }
    }

    static NodeValue Node(CombatState s, CountingRandom rng, int remaining, SolveCounters c, int widthCap, bool tight, UtilityKind u)
    {
        c.Nodes++;
        c.Check?.Invoke();

        // The key is not used, but a position the key refuses (a busy kernel queue) is refused here too.
        StateKey.Encode(s, KeyMode.Expected);
        if (Utility.SurvivalLeaf(u, s, remaining))
        {
            return NodeValue.Horizon(s, u);
        }

        var values = new List<NodeValue>();
        foreach (var m in Actions.Legal(s, Pruning.None))
        {
            values.Add(MoveValueOf(s, rng, m, remaining, c, widthCap, tight, u));
        }

        return NodeValue.Max(values, s, u);
    }

    static NodeValue MoveValueOf(CombatState s, CountingRandom rng, Move m, int remaining, SolveCounters c, int widthCap, bool tight, UtilityKind u)
    {
        tight &= u == UtilityKind.L0;
        if (Actions.EndsTurn(s, m) && remaining <= 0)
        {
            return u != UtilityKind.L0 ? NodeValue.Horizon(s, u) : tight ? NodeValue.Horizon(s, UtilityGains.Upper(s)) : NodeValue.Horizon(s);
        }

        var value = Rational.Zero;
        var mass = Rational.Zero;
        var upper = Rational.Zero;
        bool exact = true;
        foreach (var o in Successors.Expand(s, rng, m, byCard: true, widthCap, c))
        {
            var v = o.WideRoll ? (u != UtilityKind.L0 ? NodeValue.Horizon(s, u) : tight ? NodeValue.Horizon(s, UtilityGains.Upper(s)) : NodeValue.Horizon(s))
                : o.Result.Terminal ? NodeValue.End(o.State, o.Result.PlayerWon, u) : Node(o.State, o.Rng, remaining - (o.State.Turn - s.Turn), c, widthCap, tight, u);
            value += o.P * v.Value;
            mass += o.P * v.Mass;
            upper += o.P * v.Upper;
            exact &= v.Exact;
        }

        return new NodeValue(value, exact, mass, exact ? value : upper);
    }
}
